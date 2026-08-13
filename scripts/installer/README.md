# Windows bootstrapper

On a Windows machine with uv installed, freeze the bootstrapper with:
`uvx pyinstaller --onefile --console --name SquidXplorer-Setup scripts\installer\bootstrap.py`.
Ship the exe beside the squidxplorer wheel and a `uv.exe`, because a frozen exe cannot
pip-install into itself — that is why the env is private and uv-managed.
