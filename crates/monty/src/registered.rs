//! Embedder-registered source modules: the executor-side records.
//!
//! Each registered module compiles alongside the main program into its own
//! disjoint range of the VM's flat globals vector plus a zero-arg body
//! function, so `LoadGlobal`/`StoreGlobal` slot operands baked into a
//! module's bytecode resolve that module's names with no runtime
//! indirection. The `RegisteredModule` opcode executes the body on first
//! import and materializes the module object from the completed range; see
//! the VM dispatch for the runtime half and `Executor::new_with_modules`
//! for the construction pipeline.
//!
//! Sources come from the embedder, never from a filesystem: registering
//! modules adds no I/O capability to sandboxed code.

use serde::{Deserialize, Serialize};

use crate::{
    intern::{FunctionId, StringId},
    name_map::NameMap,
    value::Value,
};

/// `RegisteredModule` opcode action: an import site. Pushes the cached module
/// object, or executes the module body on first import.
pub(crate) const REGISTERED_MODULE_IMPORT: u16 = 0;

/// `RegisteredModule` opcode action: the compiler-emitted tail of a module
/// body. Builds the module object from the module's completed globals range,
/// caches it, and pushes it for the following `ReturnValue`.
pub(crate) const REGISTERED_MODULE_MAKE: u16 = 1;

/// One registered module as the executor stores it: compiled, slotted, and
/// ready for first import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RegisteredModule {
    /// The interned import name; `import <name>` resolves against this at
    /// compile time.
    pub name_id: StringId,
    /// File name shown in traceback frames, and the key the error-preview
    /// lookup matches to serve this module's source lines.
    pub file_name: String,
    /// The module's source, for traceback preview lines.
    pub source: String,
    /// Module-level names mapped to module-relative slots (0-based); combined
    /// with [`Self::globals_base`] this names every value in the module's
    /// globals range when the module object is materialized.
    pub name_map: NameMap,
    /// Offset of this module's slot range within the VM's globals vector.
    pub globals_base: u16,
    /// The zero-arg body wrapper invoked on first import.
    pub function_id: FunctionId,
}

/// One slot of the VM's per-run registered-module cache, indexed by registry
/// position.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum ModuleCacheSlot {
    /// Not imported yet; the first import executes the body.
    Absent,
    /// The body is executing right now (first import in flight). A second
    /// import observing this would mean an import cycle, which construction
    /// refuses — reaching it is an internal invariant violation, not a
    /// user-reachable state.
    InProgress,
    /// The materialized module object. Holds one strong reference; every
    /// import site pushes its own increment.
    Ready(Value),
}

/// The compile-time import-resolution registry: module names in registration
/// order. The compiler consults this after the built-in module lookup misses,
/// so a registered name can never shadow a built-in.
#[derive(Debug)]
pub(crate) struct ModuleRegistry {
    names: Vec<StringId>,
}

impl ModuleRegistry {
    pub fn new(names: Vec<StringId>) -> Self {
        Self { names }
    }

    /// Resolves an import name to its registry index.
    pub fn index_of(&self, name: StringId) -> Option<u16> {
        self.names
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| u16::try_from(index).expect("registry size is bounded at construction"))
    }
}
