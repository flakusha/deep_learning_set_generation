---
title: Piperator V0.91
subtitle: Generative Pipe Modeling for Blender
date: 2019-12-02
tags: ["plugins", "blender", "pipes"]
bigimg: [{src: "/piperator/scifi_rafinery.jpg", desc: "Science Fiction Raffinery"}]
---

Piperator is an addon for blender which helps generating
complex pipe layouts. Below are a couple examples what
can be done with this addon:

<!--more-->

{{< gallery caption-effect="fade" >}}
  {{< figure thumb="-thumb" link="/piperator/scifi_rafinery.jpg" caption="Science Fiction Refinery" >}}
  {{< figure thumb="-thumb" link="/piperator/hallway.jpg" caption="Hallways with Pipes" >}}
  {{< figure thumb="-thumb" link="/piperator/factory2.jpg" caption="Factory">}}
  {{< figure thumb="-thumb" link="/piperator/steaM_punk.jpg" caption="Steam Punk" alt="steampunk" >}}
{{< /gallery >}}

## Installation

Just download the zip file here:

https://gitlab.com/yeus/piperator/uploads/43f4ac5fd8f3eadafaa1f02174345fb3/piperator.zip

After downloading, open blender and go to:

    Edit -> Preferences -> Add-ons -> Install...

Choose the downloaded zip file and press

    "Install Add-on from File..."

Afterwards in Preferences, activate the support level
*Testing*, search for the **Piperator** plugin and
put a checkmark in the box to enable the addon.

<table>
<tr>
<td><img class="special-img-class" style="max-width:100%" src="/piperator/preferences.jpg" /> </td>
<td><img class="special-img-class" style="max-width:100%" src="/piperator/activate_piperator.jpg" /></td>
<!--![preferences](/piperator/preferences.jpg)-->
</tr></table>

## How to Use the Add-on

The Addon can be accessed through two ways in blender:

- in 3D-View: Add -> Mesh -> Piperator
- in the Sidebar (Access with shortcut key: 'N') in
  a "Piperator" tab.

![addon_pic](/piperator/plugin_components.jpg)

All generated Pipes and objects will be tracked
and can be deleted through the menu or by searching for the
operators:

In total, this Add-on adds the following operators to
blender:

Add Pipes
: Add pipes to object

Delete Pipes from Object
: Delete pipes from only the selectd object(s)

Delete All Pipes
: Delete all pipes from the current scene


### Parameters

When opening the editor the plugin has the following parameters:

![addon_pic](/piperator/operator_menu.png)

mesh mode
: selects different modes how the pipes will get generated. The different modes
  are: *Skin Modifier* (standard); *Simple Wire*; *Poly Curve*; *Pipe Meshes*;

flange probability
: the probability at which flanges will appear along the length of the pipe.
  The start and end of the pipe will always have a flange

support period
: the period at which supports get added to the pipe this is measured in blender
  units.

radius
: the radius of the pipe

material slow index
: the index of the slot of the material in the source object
  that should be used for the pipes.

resolution v
: the resolution of the pipes along their circumference

offset
: how far the pipes will be placed from the source mesh

**NEW** layer number
: Can be increased up to five layers of pipes. This makes it possible
  to have more complicated network

random seed
: get different variations for the pipes with different seed values

number of pipes
: number of pipes that should be generated

glue to surface
: if selected, pipes will prefer paths that are closer to the surface of
  the source object.

**NEW** use vertex colors
: make pipes originate from "sources" which are marked with green vertex
  colors and go to "sinks" which are marked with red color using
  vertex colors. The colors don't need to be precise, it's enough if they
  are "roughly" green or red (rgb: [>0.5,<0.5,<0.5] for red and
  (<0.5,>0.5,<0.5) for green).
  
{{< gallery caption-effect="fade" >}}
  {{< figure thumb="-thumb" link="/piperator/sink_source.jpg" 
  caption="Icosphere with Sinks (red) and Sources (green)" >}}
  {{< figure thumb="-thumb" link="/piperator/suzan_sink_source_pipes.jpg" 
  caption="Suzanne with Sinks (red) and Sources (green)" >}}
{{< /gallery >}}  
  
reset
: delete pipes from a previous operation on the source object.

debug mode
: add empties and wireframe that was used to generate the mesh.

#### mesh modes

here are some pictures showing the different mesh modes:

{{< gallery caption-effect="fade" >}}
  {{< figure thumb="-thumb" link="/piperator/skin_modifier.jpg" caption="Skin Modifier Mode" >}}
  {{< figure thumb="-thumb" link="/piperator/simple_wire.jpg" caption="Simple Wire Mode" >}}
  {{< figure thumb="-thumb" link="/piperator/poly_curve_object.jpg" caption="Poly Curve Mode">}}
  {{< figure thumb="-thumb" link="/piperator/pipe_mesh.jpg" caption="Pipe Meshes" >}}
{{< /gallery >}}

### Example

A short example on how to create a refinery-like structure:

<iframe width="560" height="315" src="https://www.youtube.com/embed/ixPworb7z4k" 
frameborder="0" allow="accelerometer; autoplay; encrypted-media;
gyroscope; picture-in-picture" allowfullscreen></iframe>

Some hints:

- pipes follow the edge of the mesh, so by influencing
  the topology of your mesh yo can define the layout of
  the pipe network.
- don't subdivide your mesh too many times, because this
  makes the pipe layout algorithm much slower
- subdivide your mesh often enough so that the pipes have
  enough edges available.
- 
