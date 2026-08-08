//! Tests for embedder-registered source modules (`MontyRun::new_with_modules`).
//!
//! Registered modules execute inside the sandbox on first import, exactly
//! once per run, each in its own namespace, with tracebacks naming the
//! module's own file.

use monty::MontyRun;
use monty_types::{CompileOptions, ExcType, MontyObject, PrintWriter, ResourceTracker, SourceModule};

fn module(name: &str, code: &str) -> SourceModule {
    SourceModule {
        name: name.to_owned(),
        file_name: format!("{name}.py"),
        code: code.to_owned(),
    }
}

fn run_with_modules(code: &str, modules: Vec<SourceModule>) -> MontyObject {
    MontyRun::new_with_modules(code.to_owned(), "main.py", vec![], modules, CompileOptions::default())
        .unwrap()
        .run_no_limits(vec![])
        .unwrap()
}

#[test]
fn import_binds_the_module_and_qualified_calls_work() {
    let networking = module("networking", "def declare(n):\n    return n + 1\n");
    let result = run_with_modules("import networking\nnetworking.declare(41)", vec![networking]);
    assert_eq!(result, MontyObject::Int(42));
}

#[test]
fn from_import_binds_attributes() {
    let networking = module("networking", "WIDTH = 7\ndef declare(n):\n    return n * WIDTH\n");
    let result = run_with_modules(
        "from networking import declare, WIDTH\ndeclare(6) - WIDTH",
        vec![networking],
    );
    assert_eq!(result, MontyObject::Int(35));
}

// Two modules with identically named helpers: each body resolves its own,
// because each module's globals occupy a disjoint slot range.
#[test]
fn same_named_helpers_stay_separate() {
    let first = module(
        "first",
        "def _helper():\n    return 10\n\ndef value():\n    return _helper()\n",
    );
    let second = module(
        "second",
        "def _helper():\n    return 20\n\ndef value():\n    return _helper()\n",
    );
    let result = run_with_modules(
        "import first\nimport second\nfirst.value() * 100 + second.value()",
        vec![first, second],
    );
    assert_eq!(result, MontyObject::Int(1020));
}

// The same module name in the main program and in a module never collide:
// each file reads its own binding.
#[test]
fn main_and_module_globals_do_not_collide() {
    let m = module("m", "x = 2\ndef read():\n    return x\n");
    let result = run_with_modules("import m\nx = 1\nx * 10 + m.read()", vec![m]);
    assert_eq!(result, MontyObject::Int(12));
}

#[test]
fn a_module_body_executes_exactly_once() {
    let m = module("m", "print(\"executed\")\nvalue = 1\n");
    let runner = MontyRun::new_with_modules(
        "import m\nfrom m import value\nimport m\nvalue".to_owned(),
        "main.py",
        vec![],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap();
    let mut output = String::new();
    let result = runner
        .run(
            vec![],
            ResourceTracker::default(),
            PrintWriter::collect_string(&mut output),
        )
        .unwrap();
    assert_eq!(result, MontyObject::Int(1));
    assert_eq!(output, "executed\n");
}

#[test]
fn a_module_imports_another_module() {
    let shared = module("shared", "BASE = 40\n");
    let networking = module(
        "networking",
        "from shared import BASE\n\ndef declare(n):\n    return BASE + n\n",
    );
    let result = run_with_modules("import networking\nnetworking.declare(2)", vec![shared, networking]);
    assert_eq!(result, MontyObject::Int(42));
}

// Import order in the registry is irrelevant: resolution is by name.
#[test]
fn registry_order_does_not_matter() {
    let networking = module(
        "networking",
        "from shared import BASE\n\ndef declare(n):\n    return BASE + n\n",
    );
    let shared = module("shared", "BASE = 40\n");
    let result = run_with_modules("import networking\nnetworking.declare(2)", vec![networking, shared]);
    assert_eq!(result, MontyObject::Int(42));
}

#[test]
fn an_import_cycle_is_refused_at_construction() {
    let a = module("a", "import b\n");
    let b = module("b", "import a\n");
    let err = MontyRun::new_with_modules(
        "import a".to_owned(),
        "main.py",
        vec![],
        vec![a, b],
        CompileOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err.exc_type(), ExcType::ImportError);
    let message = err.to_string();
    assert!(message.contains("import cycle among registered modules"), "{message}");
    assert!(message.contains("a -> b -> a"), "{message}");
}

#[test]
fn a_self_import_is_refused_at_construction() {
    let m = module("m", "import m\n");
    let err = MontyRun::new_with_modules(
        "import m".to_owned(),
        "main.py",
        vec![],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err.exc_type(), ExcType::ImportError);
    assert!(err.to_string().contains("m -> m"), "{err}");
}

// A cycle reached only through function bodies is still a cycle: the import
// executes at call time, and the dependency edges are collected everywhere.
#[test]
fn a_cycle_through_function_bodies_is_refused() {
    let a = module("a", "def f():\n    import b\n");
    let b = module("b", "def g():\n    import a\n");
    let err = MontyRun::new_with_modules(
        "import a".to_owned(),
        "main.py",
        vec![],
        vec![a, b],
        CompileOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err.exc_type(), ExcType::ImportError);
}

#[test]
fn an_unknown_import_still_raises_module_not_found() {
    let m = module("m", "value = 1\n");
    let runner = MontyRun::new_with_modules(
        "import absent".to_owned(),
        "main.py",
        vec![],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap();
    let err = runner.run_no_limits(vec![]).unwrap_err();
    assert_eq!(err.exc_type(), ExcType::ModuleNotFoundError);
}

#[test]
fn a_builtin_module_name_is_refused() {
    let err = MontyRun::new_with_modules(
        "import math".to_owned(),
        "main.py",
        vec![],
        vec![module("math", "value = 1\n")],
        CompileOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err.exc_type(), ExcType::SyntaxError);
    assert!(err.to_string().contains("collides with a built-in module"), "{err}");
}

#[test]
fn a_duplicate_module_name_is_refused() {
    let err = MontyRun::new_with_modules(
        "import m".to_owned(),
        "main.py",
        vec![],
        vec![module("m", "value = 1\n"), module("m", "value = 2\n")],
        CompileOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err.exc_type(), ExcType::SyntaxError);
    assert!(err.to_string().contains("registered twice"), "{err}");
}

#[test]
fn an_invalid_module_name_is_refused() {
    let err = MontyRun::new_with_modules(
        "1".to_owned(),
        "main.py",
        vec![],
        vec![module("not-a-name", "value = 1\n")],
        CompileOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err.exc_type(), ExcType::SyntaxError);
    assert!(err.to_string().contains("not a valid identifier"), "{err}");
}

// A traceback from inside a module names the module's own file and previews
// the module's own source line.
#[test]
fn tracebacks_name_the_module_file() {
    let networking = module("networking", "def declare(n):\n    return n / 0\n");
    let runner = MontyRun::new_with_modules(
        "import networking\nnetworking.declare(1)".to_owned(),
        "main.py",
        vec![],
        vec![networking],
        CompileOptions::default(),
    )
    .unwrap();
    let err = runner.run_no_limits(vec![]).unwrap_err();
    assert_eq!(err.exc_type(), ExcType::ZeroDivisionError);
    let message = err.to_string();
    assert!(message.contains("networking.py"), "{message}");
    assert!(message.contains("return n / 0"), "{message}");
    // The import call site in the main program is also on the stack.
    assert!(message.contains("main.py"), "{message}");
}

#[test]
fn a_dataclass_crosses_the_module_boundary() {
    let shapes = module(
        "shapes",
        "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: int\n    y: int\n",
    );
    let result = run_with_modules(
        "from shapes import Point\np = Point(x=3, y=4)\np.x * 10 + p.y",
        vec![shapes],
    );
    assert_eq!(result, MontyObject::Int(34));
}

// Module attributes are a snapshot taken when the body completes: a later
// `global` rebind inside a module function updates the module's own globals
// (its functions see the new value) but not the materialized attribute.
// Deliberate divergence from CPython's live module namespace.
#[test]
fn module_attributes_are_a_snapshot_of_the_completed_body() {
    let m = module(
        "m",
        "value = 1\n\ndef bump():\n    global value\n    value = value + 1\n    return value\n",
    );
    let result = run_with_modules("import m\nbumped = m.bump()\nbumped * 10 + m.value", vec![m]);
    assert_eq!(result, MontyObject::Int(21));
}

#[test]
fn an_import_inside_a_function_executes_the_module_once() {
    let m = module("m", "print(\"executed\")\nvalue = 5\n");
    let runner = MontyRun::new_with_modules(
        "def read():\n    import m\n    return m.value\nread() + read()".to_owned(),
        "main.py",
        vec![],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap();
    let mut output = String::new();
    let result = runner
        .run(
            vec![],
            ResourceTracker::default(),
            PrintWriter::collect_string(&mut output),
        )
        .unwrap();
    assert_eq!(result, MontyObject::Int(10));
    assert_eq!(output, "executed\n");
}

#[test]
fn a_missing_from_import_name_raises_import_error() {
    let m = module("m", "value = 1\n");
    let runner = MontyRun::new_with_modules(
        "from m import absent".to_owned(),
        "main.py",
        vec![],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap();
    let err = runner.run_no_limits(vec![]).unwrap_err();
    assert_eq!(err.exc_type(), ExcType::ImportError);
}

#[test]
fn inputs_populate_the_main_program_with_modules_present() {
    let m = module("m", "def double(n):\n    return n * 2\n");
    let runner = MontyRun::new_with_modules(
        "import m\nm.double(x)".to_owned(),
        "main.py",
        vec!["x".to_owned()],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap();
    let result = runner.run_no_limits(vec![MontyObject::Int(21)]).unwrap();
    assert_eq!(result, MontyObject::Int(42));
}

#[test]
fn dump_and_load_round_trip_with_modules() {
    let m = module("m", "def declare(n):\n    return n + 1\n");
    let runner = MontyRun::new_with_modules(
        "import m\nm.declare(41)".to_owned(),
        "main.py",
        vec![],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap();
    let bytes = runner.dump().unwrap();
    let restored = MontyRun::load(&bytes).unwrap();
    assert_eq!(restored.run_no_limits(vec![]).unwrap(), MontyObject::Int(42));
}

#[test]
fn plain_new_still_behaves_identically() {
    let runner = MontyRun::new("40 + 2".to_owned(), "main.py", vec![], CompileOptions::default()).unwrap();
    assert_eq!(runner.run_no_limits(vec![]).unwrap(), MontyObject::Int(42));
}

// Module objects are read-only: the existing `Module` type refuses
// attribute assignment, so a registered module cannot be mutated from
// outside its own body.
#[test]
fn module_attributes_refuse_assignment() {
    let m = module("m", "value = 1\n");
    let err = MontyRun::new_with_modules(
        "import m\nm.value = 2\nm.value".to_owned(),
        "main.py",
        vec![],
        vec![m],
        CompileOptions::default(),
    )
    .unwrap()
    .run_no_limits(vec![])
    .unwrap_err();
    assert_eq!(err.exc_type(), ExcType::AttributeError);
}

// Builtins resolve by name on the undefined-global path, so a plain
// (non-call) builtin reference inside module code exercises the
// per-module-range name resolution.
#[test]
fn builtins_resolve_inside_module_code() {
    let m = module("m", "def size(items):\n    f = len\n    return f(items)\n");
    let result = run_with_modules("import m\nm.size([1, 2, 3])", vec![m]);
    assert_eq!(result, MontyObject::Int(3));
}
