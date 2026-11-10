

from pathlib import Path


def configure_project_codegen():
    import torch_geometric.nn.conv.message_passing as message_passing

    original = message_passing.module_from_template
    if getattr(original, "_sregnas_codegen", False):
        return

    def module_from_template(module_name, template_path, tmp_dirname, **kwargs):
        if not module_name.startswith("sregnas."):
            return original(module_name, template_path, tmp_dirname, **kwargs)
        import importlib.util
        import sys
        import tempfile
        from jinja2 import Environment, FileSystemLoader

        if module_name in sys.modules:
            return sys.modules[module_name]
        directory = Path(__file__).resolve().parents[3] / ".runtime" / "pyg"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        env = Environment(loader=FileSystemLoader(str(Path(template_path).parent)))
        rendered = env.get_template(Path(template_path).name).render(**kwargs)
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=module_name + "_",
            suffix=".py",
            dir=directory,
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
        spec = importlib.util.spec_from_file_location(module_name, handle.name)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load generated project module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    module_from_template._sregnas_codegen = True
    message_passing.module_from_template = module_from_template
