# Deep Learning Set Generation - DLSG - Blender addon to generate sets for machine learning

This addon renders images of tubes with damages on the object and then finds this damages, marking
them for YOLO learning set format

## Plugin usage and requirements

* For this plugin to work you will need piperator addon, I have fixed minor typing error in it
and included in this repository. Install it's contents to:
    * Windows: Windows: Users\user\AppData\Roaming\Blender Foundation\Blender\3.2\addons\scripts\piperator
    * Linux: ~/.config/blender/3.2/addons/scripts/piperator
* Demo file is needed: **Tubes.blend**, this file has correct scene setup for creation of image
sets of render <-> damages on the surface
* Also Rust application is needed to be compiled, addon is installed by default to:
    * Windows: Users\user\AppData\Roaming\Blender Foundation\Blender\3.2\addons\scripts\dlsg
    * Linux: ~/.config/blender/3.2/addons/scripts/dlsg
    * Command to build application: `cargo build --release`, executable will be placed to target/release
    folder and addon will automatically use it

* After prerequisites are fulfilled open Tubes.blend and select e.g. cube (make it active),
tubes will be generated on it. Object must have three materials:
  * Material - default material for "walls" or "buildings"
  * Pipe_Support - material for pipe connections and supports
  * Pipe_Material - material for pipes with damages

* Properties tab with *Render* settings will have new panel DLSG with "Generate Image" button.
After pushing this button all the tubes will be regenerated and scene will be rendered in specified resolution.
First image is "photorealistic" and second with damages map (RGB/GS for damages and Alpha)
* After render additional Rust application will search for domains/mask components and calculate bounding boxes
and write them to *.yolo file
* All images and results are placed alongside original Blender file

## Libraries in use:
* Piperator - plugin to generate pipes using polygon centers of selected object
* Image - library used to read images in Rust

## Ways to enhance application:
* Custom pipe generator with correct tubes used in real gas/oil pipelines
* Better materials, more variants of materials
* Automatic object selection and active status assignment, loop processing
* Automatic materials and scene settings assignment
* Randomized lighting setup close to real lighting
* Tube geometry deform and bend, random material generation, more damages variants
* Better and more diverse damages, description for every damage type in *.yolo files
* Multithread domain search or any mainstream library to analyze image
* Remove warnings during Rust compilation (code refactoring)

