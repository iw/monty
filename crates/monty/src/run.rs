//! Public interface for running Monty code.
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub use monty_types::{CompileOptions, SourceModule};
use monty_types::{ExcType, MontyException, MontyObject, PrintWriter, ResourceTracker};
use ruff_python_stdlib::identifiers::is_identifier;

use crate::{
    args::Signature,
    bytecode::{Code, Compiler, FrameExit, VM},
    dump_format::{DumpKind, dump, load},
    exception_private::{ExcTypeExt, RunResult},
    expressions::Identifier,
    function::Function,
    heap::{DropWithContext, Heap, HeapReader},
    intern::{FunctionId, InternerBuilder, Interns},
    modules::StandardLib,
    name_map::NameMap,
    namespace::NamespaceId,
    object_bridge::MontyObjectExt,
    parse::{CodeRange, ParseResult, parse, parse_with_interner},
    prepare::{prepare, prepare_module_body, prepare_with_existing_names},
    registered::{ModuleRegistry, RegisteredModule},
    run_progress::{RunProgress, build_run_progress, check_snapshot_from_converted, convert_frame_exit},
    types::str::StringRepr,
    value::Value,
};

/// Primary interface for running Monty code.
///
/// `MontyRun` supports two execution modes:
/// - **Simple execution**: Use `run()` or `run_no_limits()` to run code to completion
/// - **Iterative execution**: Use `start()` to start execution which will pause at external function calls and
///   can be resumed later
///
/// # Example
/// ```
/// use monty::MontyRun;
/// use monty_types::{CompileOptions, MontyObject};
///
/// let runner = MontyRun::new(
///     "x + 1".to_owned(),
///     "test.py",
///     vec!["x".to_owned()],
///     CompileOptions::default(),
/// )
/// .unwrap();
/// let result = runner.run_no_limits(vec![MontyObject::Int(41)]).unwrap();
/// assert_eq!(result, MontyObject::Int(42));
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MontyRun {
    /// The underlying executor containing parsed AST and interns.
    executor: Executor,
}

impl MontyRun {
    /// Creates a new run snapshot by parsing the given code.
    ///
    /// This only parses and prepares the code - no heap or namespaces are created yet.
    /// Call `run_snapshot()` with inputs to start execution.
    ///
    /// # Arguments
    /// * `code` - The Python code to execute
    /// * `script_name` - The script name for error messages
    /// * `input_names` - Names of input variables
    /// * `options` - [`CompileOptions`] controlling CPython divergences; usually `CompileOptions::default()`
    ///
    /// # Errors
    /// Returns `MontyException` if the code cannot be parsed.
    pub fn new(
        code: String,
        script_name: &str,
        input_names: Vec<String>,
        options: CompileOptions,
    ) -> Result<Self, MontyException> {
        Executor::new(code, script_name, input_names, options).map(|executor| Self { executor })
    }

    /// Creates a new run snapshot with embedder-registered source modules.
    ///
    /// Each [`SourceModule`] becomes importable by its `name` from the main
    /// program and from other registered modules: `import networking` binds
    /// the module object, `from networking import declare` binds its
    /// attributes. A module body executes inside the sandbox on the first
    /// import that reaches it — exactly once per run — and its traceback
    /// frames carry the module's `file_name`. Modules see nothing of each
    /// other implicitly (each has its own namespace); built-in module names
    /// cannot be shadowed; and import cycles among registered modules are
    /// refused here, at construction.
    ///
    /// Sources come from the embedder, never from a filesystem — registering
    /// modules adds no I/O capability to sandboxed code.
    ///
    /// # Errors
    /// Returns `MontyException` if any source cannot be parsed or compiled,
    /// if a module name is invalid, duplicated, or collides with a built-in
    /// module, or if the registered imports form a cycle.
    pub fn new_with_modules(
        code: String,
        script_name: &str,
        input_names: Vec<String>,
        modules: Vec<SourceModule>,
        options: CompileOptions,
    ) -> Result<Self, MontyException> {
        Executor::new_with_modules(code, script_name, input_names, modules, options).map(|executor| Self { executor })
    }

    /// Returns the code that was parsed to create this snapshot.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.executor.code
    }

    /// Executes the code and returns both the result and reference count data, used for testing only.
    #[cfg(feature = "ref-count-return")]
    pub fn run_ref_counts(&self, inputs: Vec<MontyObject>) -> Result<RefCountOutput, MontyException> {
        self.executor.run_ref_counts(inputs)
    }

    /// Executes the code and returns reference count data while using a custom tracker, used for testing only.
    #[cfg(feature = "ref-count-return")]
    pub fn run_ref_counts_with_tracker(
        &self,
        inputs: Vec<MontyObject>,
        resource_tracker: ResourceTracker,
    ) -> Result<RefCountOutput, MontyException> {
        self.executor.run_ref_counts_with_tracker(inputs, resource_tracker)
    }

    /// Executes the code to completion assuming not external functions or snapshotting.
    ///
    /// This is marginally faster than running with snapshotting enabled since we don't need
    /// to track the position in code, but does not allow calling of external functions.
    ///
    /// # Arguments
    /// * `inputs` - Values to fill the first N slots of the namespace
    /// * `resource_tracker` - Custom resource tracker implementation
    /// * `print` - print output writer
    pub fn run(
        &self,
        inputs: Vec<MontyObject>,
        resource_tracker: ResourceTracker,
        print: PrintWriter<'_>,
    ) -> Result<MontyObject, MontyException> {
        self.executor.run(inputs, resource_tracker, print)
    }

    /// Executes the code to completion with no resource limits specified (will use the default),
    /// printing to stdout/stderr.
    pub fn run_no_limits(&self, inputs: Vec<MontyObject>) -> Result<MontyObject, MontyException> {
        self.run(inputs, ResourceTracker::default(), PrintWriter::Stdout)
    }

    /// Serializes the runner to a binary format.
    ///
    /// The serialized data can be stored and later restored with `load()`.
    /// Dumps include a format version and kind so incompatible data is rejected.
    /// This allows caching parsed code to avoid re-parsing on subsequent runs.
    ///
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn dump(&self) -> Result<Vec<u8>, postcard::Error> {
        dump(self, DumpKind::MontyRun)
    }

    /// Deserializes a runner from binary format.
    ///
    /// # Arguments
    /// * `bytes` - The serialized runner data from `dump()`
    ///
    /// # Errors
    /// Returns an error for an incompatible dump version or kind, or if
    /// deserialization fails.
    pub fn load(bytes: &[u8]) -> Result<Self, postcard::Error> {
        load(bytes, DumpKind::MontyRun)
    }

    /// Starts execution with the given inputs and resource tracker, consuming self.
    ///
    /// Creates the heap and namespaces, then begins execution.
    ///
    /// For iterative execution, `start()` consumes self and returns a `RunProgress`:
    /// - `RunProgress::FunctionCall(call)` - external function call, call `call.resume(return_value)` to resume
    /// - `RunProgress::Complete(value)` - execution finished
    ///
    /// This enables snapshotting execution state and returning control to the host
    /// application during long-running computations.
    ///
    /// # Arguments
    /// * `inputs` - Initial input values (must match length of `input_names` from `new()`)
    /// * `resource_tracker` - Resource tracker for the execution
    /// * `print` - Writer for print output
    ///
    /// # Errors
    /// Returns `MontyException` if:
    /// - The number of inputs doesn't match the expected count
    /// - An input value is invalid (e.g., `MontyObject::Repr`)
    /// - A runtime error occurs during execution
    ///
    /// # Panics
    /// This method should not panic under normal operation. Internal assertions
    /// may panic if the VM reaches an inconsistent state (indicating a bug).
    pub fn start(
        self,
        inputs: Vec<MontyObject>,
        resource_tracker: ResourceTracker,
        print: PrintWriter<'_>,
    ) -> Result<RunProgress, MontyException> {
        let executor = self.executor;

        // Create heap and VM with empty globals, then populate inputs with VM alive
        let mut heap = Heap::new(executor.namespace_size(), resource_tracker);
        let globals = executor.empty_globals();
        let (converted, vm_state) =
            HeapReader::with(&mut heap, &mut (&executor, print), |reader, (executor, print)| {
                let mut vm = VM::new(
                    globals,
                    reader,
                    &executor.interns,
                    print.reborrow(),
                    executor.assert_repr_max_bytes,
                );
                vm.set_registered(&executor.registered);
                executor.populate_inputs(inputs, &mut vm)?;

                // Start execution
                let vm_result = vm.run_module(&executor.module_code);

                // Three-phase conversion: convert while VM alive, then snapshot, then build progress
                let converted = convert_frame_exit(vm_result, &mut vm);
                let vm_state = check_snapshot_from_converted(&converted, vm);
                Ok((converted, vm_state))
            })?;
        build_run_progress(converted, vm_state, executor, heap)
    }
}

/// Lower level interface to parse code and run it to completion.
///
/// This is an internal type used by [`MontyRun`]. It stores the compiled bytecode and source code
/// for error reporting. Also used by `run_progress` and `repl` modules.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Executor {
    /// Module-level global names.
    pub(crate) globals: NameMap,
    /// Compiled bytecode for the module. Wrapped in `Arc` to avoid needing to deep clone.
    pub(crate) module_code: Arc<Code>,
    /// Interned strings used for looking up names and filenames during execution.
    pub(crate) interns: Interns,
    /// Source code for error reporting (extracting preview lines for tracebacks).
    pub(crate) code: String,
    /// Namespace slots that the REPL input-injection path writes into.
    ///
    /// Pre-resolved at snippet-construction time so the per-call hot path
    /// (`inject_inputs_into_vm`) is an O(1) slot index instead of an
    /// O(N-interns) `Interns::get_string_id_by_name` lookup per input.
    /// One entry per input value, in the order the embedder passed them.
    /// Empty for the standard (non-REPL) execution path.
    pub(crate) input_slots: Vec<NamespaceId>,
    /// UTF-8 byte cap for each operand repr in introspected assert messages.
    /// Stored with the compiled program and passed to every VM.
    pub(crate) assert_repr_max_bytes: u32,
    /// Embedder-registered source modules in registry order: compiled bodies,
    /// per-module name maps, slot bases, and sources for traceback previews.
    /// Empty for programs registered without modules.
    pub(crate) registered: Vec<RegisteredModule>,
    /// Estimated heap capacity for pre-allocation on subsequent runs.
    /// Uses AtomicUsize for thread-safety (required by PyO3's Sync bound).
    heap_capacity: AtomicUsize,
}

impl Clone for Executor {
    fn clone(&self) -> Self {
        Self {
            globals: self.globals.clone(),
            module_code: self.module_code.clone(),
            interns: self.interns.clone(),
            code: self.code.clone(),
            input_slots: self.input_slots.clone(),
            assert_repr_max_bytes: self.assert_repr_max_bytes,
            registered: self.registered.clone(),
            heap_capacity: AtomicUsize::new(self.heap_capacity.load(Ordering::Relaxed)),
        }
    }
}

impl Executor {
    /// Creates a new executor with the given code, filename, input names, and compile options.
    pub(crate) fn new(
        code: String,
        script_name: &str,
        input_names: Vec<String>,
        options: CompileOptions,
    ) -> Result<Self, MontyException> {
        Self::new_with_modules(code, script_name, input_names, Vec::new(), options)
    }

    /// Creates a new executor with embedder-registered source modules.
    ///
    /// The pipeline that gives each module its own namespace inside one flat
    /// globals vector:
    ///
    /// 1. every file parses against one chained interner (string ids are
    ///    program-unique; each file's positions carry its own file name);
    /// 2. module names are validated (identifier, unique, not a built-in) and
    ///    become the compile-time import registry;
    /// 3. the main program prepares with the input names, each module body
    ///    against a fresh empty namespace;
    /// 4. each module is assigned a disjoint slot range after the main
    ///    program's, and compiles with that base baked into every global-slot
    ///    operand, ending in a `RegisteredModule` make tail;
    /// 5. each compiled body is wrapped as a zero-arg `<module>` function so
    ///    first import runs it through the ordinary call machinery (and
    ///    suspended frames serialize like any other call);
    /// 6. the dependency edges the compiler recorded are checked for cycles.
    pub(crate) fn new_with_modules(
        code: String,
        script_name: &str,
        input_names: Vec<String>,
        modules: Vec<SourceModule>,
        options: CompileOptions,
    ) -> Result<Self, MontyException> {
        check_identifier(&input_names)?;

        // 1. Parse every file against one chained interner.
        let ParseResult {
            nodes: main_nodes,
            mut interner,
        } = parse(&code, script_name).map_err(|e| e.into_python_exc(script_name, &code))?;
        let mut parsed_modules = Vec::with_capacity(modules.len());
        for module in &modules {
            let ParseResult { nodes, interner: next } = parse_with_interner(&module.code, &module.file_name, interner)
                .map_err(|e| e.into_python_exc(&module.file_name, &module.code))?;
            parsed_modules.push(nodes);
            interner = next;
        }

        // 2. Validate names and build the import registry. `u16` bounds the
        // registry index operand.
        if modules.len() > usize::from(u16::MAX) + 1 {
            return Err(MontyException::new(
                ExcType::SyntaxError,
                Some(format!(
                    "too many registered modules ({} > {})",
                    modules.len(),
                    usize::from(u16::MAX) + 1
                )),
            ));
        }
        let mut registry_names = Vec::with_capacity(modules.len());
        for module in &modules {
            if !is_identifier(&module.name) {
                return Err(MontyException::new(
                    ExcType::SyntaxError,
                    Some(format!(
                        "Module name {} not a valid identifier",
                        StringRepr(&module.name)
                    )),
                ));
            }
            let name_id = interner.intern(&module.name);
            if StandardLib::from_string_id(name_id).is_some() {
                return Err(MontyException::new(
                    ExcType::SyntaxError,
                    Some(format!(
                        "Module name {} collides with a built-in module",
                        StringRepr(&module.name)
                    )),
                ));
            }
            if registry_names.contains(&name_id) {
                return Err(MontyException::new(
                    ExcType::SyntaxError,
                    Some(format!("Module name {} registered twice", StringRepr(&module.name))),
                ));
            }
            registry_names.push(name_id);
        }
        // Every module body wrapper carries the CPython module-frame name.
        let module_frame_name = interner.intern("<module>");
        let registry = ModuleRegistry::new(registry_names.clone());

        // 3. Prepare: the main program with its input names, each module body
        // against a fresh empty namespace (and no last-expression transform —
        // bodies must fall through to their make tail).
        let mut prepared_main = prepare(
            ParseResult {
                nodes: main_nodes,
                interner,
            },
            input_names,
        )
        .map_err(|e| e.into_python_exc(script_name, &code))?;
        let mut prepared_modules = Vec::with_capacity(modules.len());
        for (module, nodes) in modules.iter().zip(parsed_modules) {
            let prepared = prepare_module_body(ParseResult {
                nodes,
                interner: prepared_main.interner,
            })
            .map_err(|e| e.into_python_exc(&module.file_name, &module.code))?;
            prepared_main.interner = prepared.interner;
            prepared_modules.push((prepared.globals, prepared.nodes));
        }

        // 4. Disjoint slot ranges: main first (inputs keep slots 0..n), then
        // each module after the previous. The compiler re-checks per-map
        // sizes; this bounds the combined range the bases bake in.
        let mut total_namespace = prepared_main.globals.len();
        let mut bases = Vec::with_capacity(prepared_modules.len());
        for (globals, _) in &prepared_modules {
            bases.push(total_namespace);
            total_namespace += globals.len();
        }
        if total_namespace > usize::from(u16::MAX) + 1 {
            return Err(MontyException::new(
                ExcType::SyntaxError,
                Some(format!(
                    "total global namespace across the program and its registered modules ({total_namespace} slots) exceeds the bytecode limit ({})",
                    usize::from(u16::MAX) + 1
                )),
            ));
        }

        // Create interns with empty functions (functions will be set after compilation)
        let mut interns = Interns::new(prepared_main.interner, Vec::new());

        // 5. Compile: main program at base 0 with the plain tail, then each
        // module at its base with the make tail, accumulating one function
        // table so every `FunctionId` stays program-unique.
        let main_compile = Compiler::compile_module_core(
            &prepared_main.nodes,
            &interns,
            &prepared_main.globals,
            Vec::new(),
            options,
            Some(&registry),
            0,
            None,
        )
        .map_err(|e| e.into_python_exc(script_name, &code))?;
        let mut functions = main_compile.functions;

        let mut registered = Vec::with_capacity(modules.len());
        let mut import_edges = Vec::with_capacity(modules.len());
        for (index, (module, (globals, nodes))) in modules.into_iter().zip(prepared_modules).enumerate() {
            let base = u16::try_from(bases[index]).expect("bounded by the total-namespace check above");
            let tail = u16::try_from(index).expect("bounded by the registry-size check above");
            let compile = Compiler::compile_module_core(
                &nodes,
                &interns,
                &globals,
                functions,
                options,
                Some(&registry),
                base,
                Some(tail),
            )
            .map_err(|e| e.into_python_exc(&module.file_name, &module.code))?;
            functions = compile.functions;
            import_edges.push(compile.registered_imports);

            let function_index = u16::try_from(functions.len()).map_err(|_| {
                MontyException::new(
                    ExcType::SyntaxError,
                    Some("too many functions across the program and its registered modules".to_string()),
                )
            })?;
            functions.push(Function::new(
                Identifier::new(module_frame_name, CodeRange::default()),
                Signature::default(),
                0,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                0,
                false,
                compile.code,
            ));
            registered.push(RegisteredModule {
                name_id: registry_names[index],
                file_name: module.file_name,
                source: module.code,
                name_map: globals,
                globals_base: base,
                function_id: FunctionId::from_index(function_index),
            });
        }

        // 6. Refuse import cycles among registered modules, by name.
        refuse_import_cycles(&import_edges, &registered, &interns)?;

        // Set the compiled functions in the interns
        interns.set_functions(functions);

        Ok(Self {
            globals: prepared_main.globals,
            module_code: Arc::new(main_compile.code),
            interns,
            code,
            input_slots: Vec::new(),
            assert_repr_max_bytes: options.assert_message_annotations.max_bytes(),
            registered,
            heap_capacity: AtomicUsize::new(total_namespace),
        })
    }

    /// Returns the size of the module's global namespace (number of slots).
    #[inline]
    pub(crate) fn namespace_size(&self) -> usize {
        self.globals.len()
    }

    /// Returns the combined globals-vector size: the main program's namespace
    /// plus every registered module's disjoint slot range.
    #[inline]
    pub(crate) fn total_namespace_size(&self) -> usize {
        self.registered.last().map_or(self.namespace_size(), |module| {
            usize::from(module.globals_base) + module.name_map.len()
        })
    }

    /// Compiles one REPL snippet against existing session metadata.
    ///
    /// This differs from [`new`](Self::new) in three ways required for true
    /// no-replay REPL execution:
    /// - Seeds parsing from `existing_interns` so old `StringId` values stay stable.
    /// - Seeds compilation with existing functions so old `FunctionId` values remain valid.
    /// - Reuses `existing_globals` and appends new global names only.
    ///
    /// `input_names` are pre-registered in the globals map before preparation so
    /// they receive stable namespace slots that the REPL input-injection logic
    /// can use.
    pub(crate) fn new_repl_snippet(
        code: String,
        script_name: &str,
        mut existing_globals: NameMap,
        existing_interns: &Interns,
        input_names: &[String],
        options: CompileOptions,
    ) -> Result<Self, MontyException> {
        check_identifier(input_names)?;

        let mut seeded_interner = InternerBuilder::from_interns(existing_interns, &code);
        // Pre-register input names so they get stable slots before
        // preparation, and capture each input's slot index so injection
        // doesn't have to perform an O(N-interns) name→StringId scan at
        // call time (one slot per input value, in order).
        //
        // Surfaced via the standard parse/prepare error path; if the
        // embedder hands over more than `u16::MAX + 1` names the bytecode
        // encoding can't represent them all.
        let mut input_slots = Vec::with_capacity(input_names.len());
        for name in input_names {
            let name_id = seeded_interner.intern(name);
            let slot = existing_globals
                .ensure_slot(name_id, CodeRange::default())
                .map_err(|e| e.into_python_exc(script_name, &code))?;
            input_slots.push(slot);
        }

        let parse_result = parse_with_interner(&code, script_name, seeded_interner)
            .map_err(|e| e.into_python_exc(script_name, &code))?;
        let prepared = prepare_with_existing_names(parse_result, existing_globals)
            .map_err(|e| e.into_python_exc(script_name, &code))?;

        let existing_functions = existing_interns.functions_clone();
        let mut interns = Interns::new(prepared.interner, Vec::new());
        let compile_result = Compiler::compile_module_with_functions(
            &prepared.nodes,
            &interns,
            &prepared.globals,
            existing_functions,
            options,
        )
        .map_err(|e| e.into_python_exc(script_name, &code))?;
        interns.set_functions(compile_result.functions);

        Ok(Self {
            globals: prepared.globals,
            module_code: Arc::new(compile_result.code),
            interns,
            code,
            input_slots,
            assert_repr_max_bytes: options.assert_message_annotations.max_bytes(),
            registered: Vec::new(),
            heap_capacity: AtomicUsize::new(0),
        })
    }

    /// Executes the code with a custom resource tracker.
    ///
    /// This provides full control over resource tracking and garbage collection
    /// scheduling. The tracker is called on each allocation and periodically
    /// during execution to check time limits and trigger GC.
    ///
    /// # Arguments
    /// * `inputs` - Values to fill the first N slots of the namespace
    /// * `resource_tracker` - Custom resource tracker implementation
    /// * `print` - Print output writer
    fn run(
        &self,
        inputs: Vec<MontyObject>,
        resource_tracker: ResourceTracker,
        print: PrintWriter<'_>,
    ) -> Result<MontyObject, MontyException> {
        let heap_capacity = self.heap_capacity.load(Ordering::Relaxed);
        let mut heap = Heap::new(heap_capacity, resource_tracker);
        let globals = self.empty_globals();

        // Create VM first, then populate inputs with VM alive
        let result = HeapReader::with(&mut heap, &mut (self, print), |reader, (executor, print)| {
            let mut vm = VM::new(
                globals,
                reader,
                &executor.interns,
                print.reborrow(),
                executor.assert_repr_max_bytes,
            );
            vm.set_registered(&executor.registered);
            executor.populate_inputs(inputs, &mut vm)?;
            executor.run_to_completion(&mut vm)
        });

        if heap.size() > heap_capacity {
            self.heap_capacity.store(heap.size(), Ordering::Relaxed);
        }

        // Non-REPL execution has exactly one source, so every frame's filename
        // resolves to the same `self.code`.
        result.map_err(|e| e.into_python_exception(&self.interns, |filename| self.source_for(filename)))
    }

    /// Runs module code on an already-configured VM to completion.
    ///
    /// Executes [`VM::run_module`], then handles `NameLookup` and `ExternalCall`
    /// exits by raising `NameError` through the VM so tracebacks are properly
    /// captured. Finally converts the result via [`frame_exit_to_object`].
    ///
    /// This is the shared non-iterative execution core used by both the standard
    /// `run` path and the REPL's `feed_run` path.
    pub(crate) fn run_to_completion<'h>(&'h self, vm: &mut VM<'h>) -> RunResult<MontyObject> {
        let mut frame_exit_result = vm.run_module(&self.module_code);

        // Handle NameLookup and ExternalCall exits by raising NameError through the VM
        // so that traceback information is properly captured. In the non-iterative path,
        // there's no host to resolve names or external functions, so these become NameErrors.
        loop {
            match frame_exit_result {
                Ok(FrameExit::NameLookup { name_id, .. }) => {
                    let name = self.interns.get_str(name_id);
                    let err = ExcType::name_error(name);
                    frame_exit_result = vm.resume_with_exception(err.into());
                }
                Ok(FrameExit::ExternalCall {
                    function_name,
                    args,
                    name_load_ip,
                    ..
                }) => {
                    // In non-iterative execution, an ExtFunction from LoadGlobalCallable
                    // means the name was undefined — raise NameError.
                    // Restore the frame IP to the load instruction so the traceback
                    // points to the name reference, not the call expression.
                    if let Some(load_ip) = name_load_ip {
                        vm.set_instruction_ip(load_ip);
                    }
                    let name = function_name.as_str(&self.interns);
                    args.drop_with(vm);
                    let err = ExcType::name_error(name);
                    frame_exit_result = vm.resume_with_exception(err.into());
                }
                other => return frame_exit_to_object(other, vm),
            }
        }
    }

    /// Executes the code and returns both the result and reference count data, used for testing only.
    #[cfg(feature = "ref-count-return")]
    fn run_ref_counts(&self, inputs: Vec<MontyObject>) -> Result<RefCountOutput, MontyException> {
        self.run_ref_counts_with_tracker(inputs, ResourceTracker::default())
    }

    /// Executes the code and returns both the result and reference count data with a custom tracker,
    /// used for testing only.
    ///
    /// This is used for testing reference counting behavior with a custom tracker. Returns
    /// the execution result plus, in [`RefCountOutput`], a map from variable names to their
    /// reference counts (heap-allocated values only), any live-but-unreachable heap entries,
    /// and the total live heap population.
    ///
    /// For strict-matching validation, assert that `unreachable` is empty: every live heap
    /// object should be reachable from a named variable, so anything left over is a leak.
    ///
    /// Only available when the `ref-count-return` feature is enabled.
    #[cfg(feature = "ref-count-return")]
    fn run_ref_counts_with_tracker(
        &self,
        inputs: Vec<MontyObject>,
        resource_tracker: ResourceTracker,
    ) -> Result<RefCountOutput, MontyException> {
        let mut heap = Heap::new(self.namespace_size(), resource_tracker);
        let globals = self.empty_globals();

        HeapReader::with(&mut heap, &mut &*self, |reader, executor| {
            // Create VM, populate inputs, and run
            let mut vm = VM::new(
                globals,
                reader,
                &executor.interns,
                PrintWriter::Stdout,
                executor.assert_repr_max_bytes,
            );
            vm.set_registered(&executor.registered);
            executor.populate_inputs(inputs, &mut vm)?;
            let frame_exit_result = vm.run_module(&executor.module_code);

            vm.__force_gc_for_tests();

            // Take globals out of the VM so we can inspect them, but keep VM alive
            // for heap access and later conversion.
            let globals = vm.take_globals();

            // Read refcounts BEFORE converting the return value, because
            // `frame_exit_to_object` drops the return value (decrementing its refcount).
            let mut counts = ahash::AHashMap::new();
            let mut roots = Vec::new();

            for (namespace_id, name_id) in executor.globals.iter() {
                let idx = namespace_id.index();
                if idx < globals.len()
                    && let Value::Ref(id) = &globals[idx]
                {
                    counts.insert(executor.interns.get_str(name_id).to_owned(), vm.heap.get_refcount(*id));
                    roots.push(*id);
                }
            }
            // The module's result is a root too: it is still owned by the pending
            // `FrameExit::Return` here, since `frame_exit_to_object` below is what drops it.
            if let Ok(FrameExit::Return(Value::Ref(id))) = &frame_exit_result {
                roots.push(*id);
            }
            // Those are the only roots: locals are gone once the module frame exits, so
            // anything still live must hang off a name or the result to not be a leak.
            let unreachable: Vec<String> = vm
                .heap
                .unreachable_entries(roots)
                .into_iter()
                .map(|(id, ty)| format!("{} (id {})", ty.name(vm.heap, &executor.interns), id.index()))
                .collect();
            let heap_count = vm.heap.entry_count();

            // Convert return value while VM is still alive (needs access to interns).
            // Non-REPL: single source, so every frame resolves to `executor.code`.
            let py_object = frame_exit_to_object(frame_exit_result, &mut vm)
                .map_err(|e| e.into_python_exception(&executor.interns, |filename| executor.source_for(filename)))?;

            // Drop globals with proper ref counting
            globals.drop_with(vm.heap);

            let allocations_since_gc = vm.heap.get_allocations_since_gc();

            Ok(RefCountOutput {
                py_object,
                counts,
                unreachable,
                heap_count,
                allocations_since_gc,
            })
        })
    }

    /// Creates an empty globals vector with all slots set to `Undefined`.
    ///
    /// Used to initialize global storage before input population. The VM is created
    /// with these empty globals, then [`populate_inputs`](Self::populate_inputs) fills
    /// the input slots while the VM is alive.
    pub(crate) fn empty_globals(&self) -> Vec<Value> {
        // The one flat vector spans the main program's namespace and every
        // registered module's disjoint slot range.
        (0..self.total_namespace_size()).map(|_| Value::Undefined).collect()
    }

    /// Serves the source text for a traceback frame's file name: a registered
    /// module's source for its file, the main program's for everything else.
    pub(crate) fn source_for(&self, filename: &str) -> Option<&str> {
        self.registered
            .iter()
            .find(|module| module.file_name == filename)
            .map(|module| module.source.as_str())
            .or(Some(self.code.as_str()))
    }

    /// Converts `MontyObject` inputs to `Value`s and writes them into the VM's globals.
    ///
    /// This runs with the VM alive so that `to_value` has access to the full VM context.
    /// On error partway through, the VM's `Drop` impl will drain globals and
    /// properly decrement refcounts for any already-converted values.
    pub(crate) fn populate_inputs(&self, inputs: Vec<MontyObject>, vm: &mut VM<'_>) -> Result<(), MontyException> {
        if inputs.len() > self.namespace_size() {
            return Err(MontyException::runtime_error("too many inputs for namespace"));
        }
        for (i, input) in inputs.into_iter().enumerate() {
            let value = input
                .to_value(vm)
                .map_err(|e| MontyException::runtime_error(format!("invalid input type: {e}")))?;
            vm.globals[i] = value;
        }
        Ok(())
    }
}

/// Converts module/frame exit results into plain `MontyObject` outputs.
///
/// Used by non-iterative execution paths where suspendable outcomes (external calls,
/// name lookups) are not supported and should produce errors.
pub(crate) fn frame_exit_to_object(frame_exit_result: RunResult<FrameExit>, vm: &mut VM<'_>) -> RunResult<MontyObject> {
    match frame_exit_result? {
        FrameExit::Return(return_value) => Ok(MontyObject::new(return_value, vm)),
        FrameExit::ExternalCall {
            function_name, args, ..
        } => {
            args.drop_with(vm);
            let function_name = function_name.as_str(vm.interns);
            Err(ExcType::not_implemented(format!(
                "External function '{function_name}' not implemented with standard execution"
            ))
            .into())
        }
        FrameExit::OsCall { function_call, .. } => {
            let name = function_call.name();
            function_call.drop_with(vm);
            Err(
                ExcType::not_implemented(format!("OS function '{name}' not implemented with standard execution"))
                    .into(),
            )
        }
        FrameExit::MethodCall { method_name, args, .. } => {
            args.drop_with(vm);
            let name = method_name.as_str(vm.interns);
            Err(
                ExcType::not_implemented(format!("Method call '{name}' not implemented with standard execution"))
                    .into(),
            )
        }
        FrameExit::ResolveFutures(_) => {
            Err(ExcType::not_implemented("async futures not supported by standard execution.").into())
        }
        FrameExit::NameLookup { name_id, .. } => {
            let name = vm.interns.get_str(name_id);
            Err(ExcType::name_error(name).into())
        }
    }
}

/// Output from `run_ref_counts` containing reference count and heap information.
///
/// Used for testing GC behavior and reference counting correctness.
#[cfg(feature = "ref-count-return")]
#[derive(Debug)]
pub struct RefCountOutput {
    pub py_object: MontyObject,
    pub counts: ahash::AHashMap<String, usize>,
    /// Live heap entries reachable from no named variable, described as
    /// `"<type> (id N)"`. Non-empty means the run leaked: a missed `drop_with`
    /// left an object alive that nothing can reach. Reachability is transitive,
    /// so objects owned by another object are accounted for by their owner.
    pub unreachable: Vec<String>,
    pub heap_count: usize,
    /// Number of GC-tracked allocations since the last cycle collection.
    ///
    /// If the collector ran during execution, this will be much lower than
    /// the total number of GC-tracked allocations performed. Compare against
    /// the configured `gc_interval` to verify GC fired at the expected
    /// cadence.
    pub allocations_since_gc: u32,
}

/// Check if input names are valid Python identifiers.
///
/// `is_identifier` also checks that the names are not keywords.
fn check_identifier(input_names: &[String]) -> Result<(), MontyException> {
    for name in input_names {
        if !is_identifier(name) {
            return Err(MontyException::new(
                ExcType::SyntaxError,
                Some(format!("Input name {} not a valid identifier", StringRepr(name))),
            ));
        }
    }
    Ok(())
}

/// Refuses import cycles among registered modules, naming the cycle path.
///
/// `edges[i]` lists the registry indexes module `i` imports anywhere in its
/// body (function and class bodies included) — recorded by the compiler while
/// it resolved the imports. The main program cannot participate in a cycle
/// because it is not importable, so only module-to-module edges are walked.
/// With the graph acyclic, first-import execution recurses cleanly:
/// importing a module first executes its own imports, depth-first.
fn refuse_import_cycles(
    edges: &[Vec<u16>],
    registered: &[RegisteredModule],
    interns: &Interns,
) -> Result<(), MontyException> {
    const UNVISITED: u8 = 0;
    const IN_STACK: u8 = 1;
    const DONE: u8 = 2;

    fn visit(node: usize, edges: &[Vec<u16>], marks: &mut [u8], stack: &mut Vec<usize>) -> Option<Vec<usize>> {
        marks[node] = IN_STACK;
        stack.push(node);
        for &next in &edges[node] {
            let next = usize::from(next);
            match marks[next] {
                IN_STACK => {
                    // The cycle is the stack from `next`'s frame down, closed
                    // back onto `next`.
                    let start = stack
                        .iter()
                        .position(|&n| n == next)
                        .expect("in-stack node is on the stack");
                    let mut cycle = stack[start..].to_vec();
                    cycle.push(next);
                    return Some(cycle);
                }
                UNVISITED => {
                    if let Some(cycle) = visit(next, edges, marks, stack) {
                        return Some(cycle);
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        marks[node] = DONE;
        None
    }

    let mut marks = vec![UNVISITED; edges.len()];
    let mut stack = Vec::new();
    for node in 0..edges.len() {
        if marks[node] == UNVISITED
            && let Some(cycle) = visit(node, edges, &mut marks, &mut stack)
        {
            let path: Vec<&str> = cycle
                .iter()
                .map(|&index| interns.get_str(registered[index].name_id))
                .collect();
            return Err(MontyException::new(
                ExcType::ImportError,
                Some(format!("import cycle among registered modules: {}", path.join(" -> "))),
            ));
        }
    }
    Ok(())
}
