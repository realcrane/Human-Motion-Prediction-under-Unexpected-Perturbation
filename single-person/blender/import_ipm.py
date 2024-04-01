import bpy
import numpy as np
import os
c3ds_path = 'c3d_ipm/ipm_gr_pred/exp1/'
points_path = 'points_ipm/ipm_gr_pred/exp1/'

c3ds = os.listdir(c3ds_path)
ball_size = 0.05

armature_obj = None

print('Matching files: ' + str(len(c3ds)))
if len(c3ds) == 0:
    raise Exception('No matching files found')
cube_len = 0.02 #m

for file in c3ds:
    # Parse
    name = file[:-4]
    c3d_path = c3ds_path + file
    bpy.ops.import_anim.c3d(filepath=c3d_path, print_file=False)
    point_path = points_path + file[:-3] + 'npy'
    points = np.load(point_path)
    cart_h = points[0,0,1]
    # Fetch loaded objects
    obj = bpy.context.selected_objects[0]
    action = obj.animation_data.action

    # Add a cube
    bpy.ops.mesh.primitive_cube_add(size=cart_h, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    cube = bpy.context.selected_objects[0]
    cube.parent = obj
    cube.parent_type = 'BONE'
    cube.parent_bone = 'cart'

    # Add a rod
    bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    arm = bpy.context.selected_objects[0]
    arm.parent = obj
    arm.parent_type = 'BONE'
    arm.parent_bone = 'cart'
    arm.scale = (0.3, 1.0, 0.3)
    bpy.ops.object.editmode_toggle()
    # arm.data.edit_bones.keys()
    tail = (points[0, 1, :] - points[0, 0, :])
    tail[1] = tail[1] - np.sign(tail[1]) * cube_len
    arm.data.edit_bones['Bone'].tail = tail
    bpy.ops.object.posemode_toggle()
    bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
    arm.pose.bones["Bone"].constraints["Damped Track"].target = obj
    arm.pose.bones["Bone"].constraints["Damped Track"].subtarget = 'rod'
    bpy.ops.object.posemode_toggle()

    # Add a ball
    bpy.ops.mesh.primitive_uv_sphere_add(radius = ball_size, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    ball = bpy.context.selected_objects[0]
    ball.parent = obj
    ball.parent_type = 'BONE'
    ball.parent_bone = 'rod'

