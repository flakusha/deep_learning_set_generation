import os, bpy, time, random, subprocess
from bpy.types import Operator
from bpy.utils import register_class, unregister_class

class DLSG_OT_Main(Operator):
    """Creates image and traces it into YOLO format"""
    bl_idname = "dlsg.dlsg_generate_set"
    bl_label = "Generate Image"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        time_generation = time.time()
        generate_set(context, str(int(time_generation)))
        time_generation_f = (time.time() - time_generation) / 60

        print("Info: Set generation took {:.3f} minutes"\
            .format(time_generation_f))

        return {"FINISHED"}

def register():
    register_class(DLSG_OT_Main)

def unregister():
    unregister_class(DLSG_OT_Main)

def generate_set(context, image_name):
    """Creates pipes for active object, renders scene and processes image.
    Scene must be setup and camera too."""
    active_object = context.view_layer.objects.active
    file_cur = bpy.path.abspath("//")
    print("Debug: Current Blender3D file:", file_cur)

    # Clean up scene from objects present in previous frame, except active
    # object and camera
    for obj in bpy.data.objects:
        if obj != active_object and obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj)

    purge_orphans()

    # Piperator addon operator with default settings, except some
    bpy.ops.mesh.add_pipes(
        mode = "skin",
        flange_appearance = 0.3,
        support_period = 0.3,
        radius = 0.05,
        material_idx = 2, # Pipe_Material
        res_v = 16, # non default
        offset = 0.11,
        offset_num = 2,
        seed = random.randint(0, 1_000_000_000),
        number = 30,
        surfaceglue = True,
        use_vertex_colors = False,
        reset = True,
        debug = 0,
    )

    # Assign correct materials
    processed_meshes = set()
    for obj in context.scene.objects:
        if obj.type == "MESH":
            context.view_layer.objects.active = obj
            if obj.data not in processed_meshes:
                if obj.name.startswith(("clamp", "pipe_objects")):
                    if len(obj.data.materials) > 0:
                        obj.data.materials[0] =\
                            bpy.data.materials["Pipe_Support"]
                    else:
                        obj.data.materials.append(
                            bpy.data.materials["Material"]
                        )
                        obj.data.materials.append(
                            bpy.data.materials["Pipe_Support"]
                        )
                        obj.data.materials.append(
                            bpy.data.materials["Pipe_Material"]
                        )


            if obj != active_object:
                bpy.ops.object.shade_smooth()
                obj.data.use_auto_smooth = True

            processed_meshes.add(obj.data)

    # Rename output image, so old results are not rewritten
    tree = context.scene.node_tree
    node_out = tree.nodes["File Output"]
    
    # Base path is //, so images are saved in the folder of blender file
    # time is written after default slot path name (render or damage)
    # render is just an image, and damage is AOV from material
    dmg_path = None
    for slot in node_out.file_slots:
        if len(slot.path.split()) == 0:
            slot_path = "{}_{}".format(slot.path, image_name)
        else:
            slot_path = "{}_{}".format(slot.path.split()[0], image_name)

        slot.path = slot_path

        if "damage" in slot_path:
            dmg_path = "{}{}.png".format(
                os.path.join(file_cur, slot_path),
                str(bpy.context.scene.frame_current).zfill(4),
            )

    # Now the scene is ready for render
    bpy.ops.render.render()

    # Run Rust standalone binary in a separate process
    if os.name == "posix":
        rust_app_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))), "target", "release",
                    "deep_learning_set_generation"
        )
    elif os.name == "nt":
        rust_app_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))), "target", "release",
                    "deep_learning_set_generation.exe"
        )

    subprocess.Popen(args = (rust_app_path, dmg_path))

def purge_orphans():
    """Clean up file from deleted assets."""
    for block in bpy.data.objects:
        if block.users == 0:
            bpy.data.objects.remove(block)

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
