import os

SETTINGS_PRIORITY = 1

PYFORMS_STYLESHEET = 'style.css'
if not os.path.exists(PYFORMS_STYLESHEET):
    PYFORMS_STYLESHEET = 'GUI\style.css'

PYFORMS_STYLESHEET_LINUX = 'style.css'