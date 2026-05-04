"""
main.py
=======
AQuant entry point.

Imports from app_factory, state_machine, websocket_manager.
All heavy logic moved to dedicated modules in v4.4 architecture refactor.
"""
from app.app_factory import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
