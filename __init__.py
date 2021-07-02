import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "mod"))

import bpy
# from mathutils import Color
# from bpy.types import PointerProperty
from bpy.utils import register_class, unregister_class

# from . import preferences
from mod import gen_tubes, ui

classes = (
    # preferences.dlsg,
    ui.DLSG_PT_UI,
    gen_tubes.DLSG_OT_Main,
)

bl_info = {
    "name": "Deep Learning Set Generation",
    "description": "Plugin creates tubes sets for machine vision",
    "author": "Constantine Fedotov <zenflak@gmail.com>",
    "version": (0, 0, 1),
    "blender": (2, 93, 0),
    "category": "Render",
    "location": "Properties > Render > DLSG",
}

def register():
    for cls in classes:
        register_class(cls)

def unregister():
    for cls in reversed(classes):
        unregister_class(cls)
