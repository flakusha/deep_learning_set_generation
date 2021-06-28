import bpy
from bpy.types import Panel
from bpy.utils import register_class, unregister_class

class DLSG_PT_UI(Panel):
    bl_label = "DLSG Panel"
    bl_idname = "DLSG_PT_UI"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.scale_y = 2.0
        row.operator("dlsg.dlsg_generate_set")

def register():
    register_class(DLSG_PT_UI)

def unregister():
    unregister_class(DLSG_PT_UI)
