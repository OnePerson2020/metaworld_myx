# 项目导出

**文件数量**: 8  
**总大小**: 31.2 KB  
**Token 数量**: 8.5K  
**生成时间**: 2025/7/30 11:12:20

## 文件结构

```
📁 .
  📁 ppo_test
    📁 xml
      📄 basic_scene.xml
      📄 peg_block copy.xml
      📄 peg_block_dependencies.xml
      📄 peg_block.xml
      📄 peg_insert_dependencies.xml
      📄 sawyer_peg_insertion_side.xml
      📄 xyz_base_dependencies.xml
      📄 xyz_base.xml
```

## 源文件

### ppo_test/xml/basic_scene.xml

*大小: 3.1 KB | Token: 807*

```xml
<mujocoinclude>
    <option timestep='0.0025' iterations="150" tolerance="1e-10" solver="Newton" jacobian="dense" cone="elliptic"/>

    <asset>
        <!-- night sky -->
        <!-- <texture name="skybox" type="skybox" builtin="gradient" rgb1=".08 .09 .10" rgb2="0 0 0"
               width="800" height="800" mark="random" markrgb=".8 .8 .8"/> -->
        <texture type="skybox" builtin="gradient" rgb1=".50 .495 .48" rgb2=".50 .495 .48" width="32" height="32"/>
        <texture name="T_table" type="cube" file="./textures/wood2.png"/>
        <texture name="T_floor" type="2d" file="./textures/floor2.png"/>

        <material name="basic_floor" texture="T_floor" texrepeat="12 12" shininess=".3" specular="0.5"
                  reflectance="0.2"/>
        <material name="table_wood" texture="T_table" shininess=".3" specular="0.5"/>
        <material name="table_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0"/>

        <mesh file="./table/tablebody.stl" name="tablebody" scale="1 1 1"/>
        <mesh file="./table/tabletop.stl" name="tabletop" scale="1 1 1"/>
    </asset>

    <asset>
        <texture name="T_wallmetal" type="cube" file="./textures/metal.png"/>
        <material name="wall_metal" texture="T_wallmetal" shininess="1" reflectance="1" specular=".5"/>
    </asset>

    <visual>
        <map fogstart="1.5" fogend="5" force="0.1" znear="0.01"/>
        <quality shadowsize="4096" offsamples="4"/>

        <headlight ambient="0.4 0.4 0.4"/>

    </visual>

    <worldbody>
        <light castshadow="false" directional='true' diffuse='.3 .3 .3' specular='0.3 0.3 0.3' pos='-1 -1 1'
               dir='1 1 -1'/>
        <light directional='true' diffuse='.3 .3 .3' specular='0.3 0.3 0.3' pos='1 -1 1' dir='-1 1 -1'/>
        <light castshadow="false" directional='true' diffuse='.3 .3 .3' specular='0.3 0.3 0.3' pos='0 1 1'
               dir='0 -1 -1'/>
        <body name="tablelink" pos="0 .6 0">
            <geom material="table_wood" group="1" type="box" size=".7 .4 .027" pos="0 0 -.027" conaffinity="0"
                  contype="0"/>
            <geom material="table_wood" group="1" mesh="tablebody" pos="0 0 -0.65" type="mesh" conaffinity="0"
                  contype="0"/>
            <geom material="table_col" group="4" pos="0.0 0.0 -0.46" size="0.7 0.4 0.46" type="box" conaffinity="1"
                  contype="0"/>
        </body>

        <body name="RetainingWall" pos="0.0 0.6 0.06">
            <geom material="wall_metal" type="box" size=".7 .01 .06" pos="0. -0.39 0." conaffinity="1" condim="3"
                  contype="0"/>
            <geom material="wall_metal" type="box" size=".7 .01 .06" pos="0. 0.39 0." conaffinity="1" condim="3"
                  contype="0"/>
            <geom material="wall_metal" type="box" size=".01 .38 .06" pos="-.69 0. 0." conaffinity="1" condim="3"
                  contype="0"/>
            <geom material="wall_metal" type="box" size=".01 .38 .06" pos=".69 0. 0." conaffinity="1" condim="3"
                  contype="0"/>
        </body>

        <geom name="floor" size="4 4 .1" pos="0 0 -.913" conaffinity="1" contype="1" type="plane" material="basic_floor"
              condim="3"/>

    </worldbody>

</mujocoinclude>
```

### ppo_test/xml/peg_block copy.xml

*大小: 1.3 KB | Token: 340*

```xml
<mujocoinclude>
    <body childclass="peg_block_base">
      <geom material="peg_block_red" mesh="block_inner" pos="0 0 0.095"/>
      <geom material="peg_block_wood" mesh="block_outer" pos="0 0 0.1"/>
      <geom class="peg_block_col" pos="0 0 0.195" size="0.09 0.1 0.005" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0 0 0.05" size="0.09 0.096 0.05" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="-0.06 0 0.13" size="0.03 0.096 0.03" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0.06 0 0.13" size="0.03 0.096 0.03" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0 0 0.175" size="0.09 0.096 0.015" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0.095 0 0.1" size="0.005 0.1 0.1" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="-0.095 0 0.1" size="0.005 0.1 0.1" type="box" mass="1000"/>
      <site name="hole" pos="0 -.096 0.13" size="0.005" rgba="0 0.8 0 1"/>
      <site name="bottom_right_corner_collision_box_1" pos="0.1 -0.11 0.01" size="0.0001"/>
      <site name="top_left_corner_collision_box_1" pos="-0.1 -.15 0.096" size="0.0001"/>
      <site name="bottom_right_corner_collision_box_2" pos="0.1 -0.11 0.16" size="0.0001"/>
      <site name="top_left_corner_collision_box_2" pos="-0.1 -.17 0.19" size="0.0001"/>
    </body>
</mujocoinclude>
```

### ppo_test/xml/peg_block_dependencies.xml

*大小: 1.2 KB | Token: 317*

```xml
<mujocoinclude>
    <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
    <asset>
      <texture name="T_peg_block_wood" type="cube" file="./textures/wood1.png"/>

      <material name="peg_block_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0"/>
      <material name="peg_block_wood" texture="T_peg_block_wood" shininess="1" reflectance=".7" specular=".5"/>
      <material name="peg_block_red" rgba=".55 0 0 1" shininess="1" reflectance=".7" specular=".5"/>

    </asset>
    <default>

      <default class="peg_block_base">
          <joint armature="0.001" damping="2" limited="true"/>
          <geom conaffinity="0" contype="0" group="1" type="mesh"/>
          <position ctrllimited="true" ctrlrange="0 1.57"/>
          <default class="peg_block_viz">
              <geom condim="4" type="mesh"/>
          </default>
          <default class="peg_block_col">
            <geom conaffinity="1" condim="3" contype="1" group="4" material="peg_block_col" solimp="0.8 0.95 0.001" solref="0.1 1"/>
          </default>
      </default>
    </default>

    <asset>
      <mesh file="./peg_block/block_inner.stl" name="block_inner"/>
        <mesh file="./peg_block/block_outer.stl" name="block_outer"/>
    </asset>

</mujocoinclude>
```

### ppo_test/xml/peg_block.xml

*大小: 2.3 KB | Token: 585*

```xml
<mujocoinclude>
      <body childclass="peg_block_base">
            <geom material="peg_block_red" mesh="block_inner" pos="0 0 0.095"/>
            <geom material="peg_block_wood" mesh="block_outer" pos="0 0 0.1"/>

            <geom class="peg_block_col" pos="0 0 0.195" size="0.09 0.1 0.005" type="box" mass="1000"/>
            <geom class="peg_block_col" pos="0.095 0 0.1" size="0.005 0.1 0.1" type="box" mass="1000"/>
            <geom class="peg_block_col" pos="-0.095 0 0.1" size="0.005 0.1 0.1" type="box" mass="1000"/>

            <geom class="peg_block_col" pos="0 0.01 0.05" size="0.09 0.086 0.05" type="box" mass="1000"/>
            <geom class="peg_block_col" pos="-0.06 0.01 0.13" size="0.03 0.086 0.03" type="box" mass="1000"/>
            <geom class="peg_block_col" pos="0.06 0.01 0.13" size="0.03 0.086 0.03" type="box" mass="1000"/>
            <geom class="peg_block_col" pos="0 0.01 0.175" size="0.09 0.086 0.015" type="box" mass="1000"/>

            <!-- <geom type="box" conaffinity="1" contype="1" group="1" material="peg_block_col"
                  size="0.03 0.017071 0.005"
                  pos="0 -0.101 0.175"
                  euler="-45 0 0"/>

            <geom type="box" conaffinity="1" contype="1" group="4" material="peg_block_col"
                  size="0.03 0.017071 0.005"
                  pos="0 -0.101 0.085"
                  euler="45 0 0"/>

            <geom type="box" conaffinity="1" contype="1" group="4" material="peg_block_col"
                  size="0.017071 0.005 0.03"
                  pos="-0.042 -0.11 0.13"
                  euler="0 0 45"/>

            <geom type="box" conaffinity="1" contype="1" group="4" material="peg_block_col"
                  size="0.017071 0.005 0.03"
                  pos="0.042 -0.11 0.13"
                  euler="0 0 -45"/> -->



            <site name="hole" pos="0 -.096 0.13" size="0.005" rgba="0 0.8 0 1"/>
            <site name="bottom_right_corner_collision_box_1" pos="0.1 -0.11 0.01" size="0.0001"/>
            <site name="top_left_corner_collision_box_1" pos="-0.1 -.15 0.096" size="0.0001"/>
            <site name="bottom_right_corner_collision_box_2" pos="0.1 -0.11 0.16" size="0.0001"/>
            <site name="top_left_corner_collision_box_2" pos="-0.1 -.17 0.19" size="0.0001"/>
      </body>
      </mujocoinclude>
```

### ppo_test/xml/peg_insert_dependencies.xml

*大小: 1006 B | Token: 252*

```xml
<mujocoinclude>
    <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
    <asset>
      <texture name="T_peg_wood" type="cube" file="./textures/wood1.png"/>

      <material name="peg_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0"/>
      <material name="peg_green" rgba="0 .5 0 1" shininess="1" reflectance=".7" specular=".5"/>
      <material name="peg_black" rgba=".15 .15 .15 1" shininess="1" reflectance=".7" specular=".5"/>
      <material name="peg_wood" rgba=".55 .55 .55 1" texture="T_peg_wood" shininess="1" reflectance=".7" specular=".5"/>
    </asset>
    <default>
      <default class="peg_base">
          <!-- <joint armature="0.001" damping="2"/> -->
          <geom conaffinity="0" contype="0" group="1" type="mesh"/>
          <default class="peg_col">
              <geom conaffinity="1" condim="3" contype="1" group="4" material="peg_col" solimp="0.99 0.99 0.01" solref="0.01 1"/>
          </default>

      </default>
    </default>

</mujocoinclude>
```

### ppo_test/xml/sawyer_peg_insertion_side.xml

*大小: 1.8 KB | Token: 473*

```xml
<mujoco>
    <include file="./basic_scene.xml"/>
    <include file="./peg_block_dependencies.xml"/>
    <include file="./peg_insert_dependencies.xml"/>
    <include file="./xyz_base_dependencies.xml"/>

    <worldbody>
      <include file="./xyz_base.xml"/>

        <body name="peg" pos="0 0.6 0.03">
          <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
          <geom name="peg" euler="0 1.57 0" size="0.015 0.015 0.12" type="box" mass=".1" rgba="0.3 1 0.3 1" conaffinity="1" contype="1" group="1" solimp="0.95 0.99 0.01" solref="0.01 1"/>
          <joint type="free" limited="false" damping="0.005"/>
          <!-- <geom name="pegHead_geom" type="box" size="0.005 0.016 0.016" pos="-0.12 0 0" mass=".001" rgba="1 0 0 0.5" conaffinity="1" contype="1" group="1"/> -->
          <site name="pegHead" pos="-0.12 0 0" size="0.005" rgba="0.8 0 0 1"/>
          <!-- <geom name="pegEnd_geom" type="box" size="0.005 0.016 0.016" pos="0.12 0 0" mass=".001" rgba="1 0 0 0.5" conaffinity="1" contype="1" group="1"/> -->
          <site name="pegEnd" pos="0.12 0 0" size="0.005" rgba="0.8 0 0 1"/>
          <site name="pegGrasp" pos=".0 .0 .0" size="0.005" rgba="0.8 0 0 1"/>
        </body>

        <body name="box" euler="0 0 1.57" pos="-0.3 0.6 0">
          <include file="./peg_block.xml"/>
        </body>
        <site name="goal" pos="0 0.6 0.05" size="0.01" rgba="0.8 0 0 0"/>

    </worldbody>

    <actuator>
        <position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="1000"  user="1"/>
        <position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="1000"  user="1"/>
    </actuator>

    <equality>
        <weld body1="mocap" body2="hand" solref="0.001 1"></weld>
    </equality>
    
    <sensor>
        <force name="peg_force_sensor" site="pegHead"/>
        <torque name="peg_torque_sensor" site="pegHead"/>
    </sensor>

</mujoco>
```

### ppo_test/xml/xyz_base_dependencies.xml

*大小: 1.4 KB | Token: 366*

```xml
<mujocoinclude>
    <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
    <asset>

      <material name="xyz_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0.5"/>

      <mesh file="./xyz_base/base.stl" name="base"/>
      <mesh file="./xyz_base/eGripperBase.stl" name="eGripperBase"/>
      <mesh file="./xyz_base/head.stl" name="head"/>
      <mesh file="./xyz_base/l0.stl" name="l0"/>
      <mesh file="./xyz_base/l1.stl" name="l1"/>
      <mesh file="./xyz_base/l2.stl" name="l2"/>
      <mesh file="./xyz_base/l3.stl" name="l3"/>
      <mesh file="./xyz_base/l4.stl" name="l4"/>
      <mesh file="./xyz_base/l5.stl" name="l5"/>
      <mesh file="./xyz_base/l6.stl" name="l6"/>
      <mesh file="./xyz_base/pedestal.stl" name="pedestal"/>
    </asset>

    <default>

      <default class="xyz_base">
          <joint armature="0.001" damping="2" limited="true"/>
          <geom conaffinity="0" contype="0" group="1" type="mesh"/>
          <position ctrllimited="true" ctrlrange="0 1.57"/>
          <default class="base_viz">
              <geom conaffinity="0" condim="4" contype="0" group="1" margin="0.001" solimp=".8 .9 .01" solref=".02 1" type="mesh"/>
          </default>
          <default class="base_col">
              <geom conaffinity="1" condim="4" contype="1" group="4" margin="0.001" material="xyz_col" solimp=".8 .9 .01" solref=".02 1"/>
          </default>
      </default>
    </default>

</mujocoinclude>
```

### ppo_test/xml/xyz_base.xml

*大小: 19.0 KB | Token: 5.3K*

```xml
<mujocoinclude>
  <!--
  Usage:

  <mujoco>
  	<compiler meshdir="../meshes/sawyer" ...></compiler>
  	<include file="shared_config.xml"></include>
      (new stuff)
  	<worldbody>
  		<include file="sawyer_xyz_base.xml"></include>
          (new stuff)
  	</worldbody>
  </mujoco>
  -->

      <camera pos="0 0.5 1.5" name="topview" />
      <camera name="corner" mode="fixed" pos="-1.1 -0.4 0.6" xyaxes="-1 1 0 -0.2 -0.2 -1"/>
      <camera name="corner2" fovy="60" mode="fixed" pos="1.3 -0.2 1.1" euler="3.9 2.3 0.6"/>
      <camera name="corner3" fovy="45" mode="fixed" pos="0.9 0 1.5" euler="3.5 2.7 1"/>
      <!--<geom name="floor" type="plane" pos="0 0 -.9" size="10 10 10"-->
            <!--rgba="0 0 0 1" contype="15" conaffinity="15" />-->
      <!--<geom name="tableTop" type="box" pos="0 0.6 -0.45" size="0.4 0.2 0.45"
            rgba=".6 .6 .5 1" contype="15" conaffinity="15" />-->
      <!-- <geom name="tableTop" type="plane" pos="0 0.6 0" size="0.4 0.4 0.5" -->
            <!-- rgba=".6 .6 .5 1" contype="1" conaffinity="1" friction="2 0.1 0.002" material="light_wood_v3"/> -->

      <body name="base" childclass="xyz_base" pos="0 0 0">
          <site name="basesite" pos="0 0 0" size="0.01" />
          <inertial pos="0 0 0" mass="0" diaginertia="0 0 0" />
          <body name="controller_box" pos="0 0 0">
              <inertial pos="-0.325 0 -0.38" mass="46.64" diaginertia="1.71363 1.27988 0.809981" />
              <geom size="0.11 0.2 0.265" pos="-0.325 0 -0.38" type="box" rgba="0.2 0.2 0.2 1"/>
          </body>
          <body name="pedestal_feet" pos="0 0 0">
              <inertial pos="-0.1225 0 -0.758" mass="167.09" diaginertia="8.16095 9.59375 15.0785" />
              <geom size="0.385 0.35 0.155" pos="-0.1225 0 -0.758" type="box" rgba="0.2 0.2 0.2 1"
                    contype="0"
                    conaffinity="0"
              />
          </body>
          <body name="torso" pos="0 0 0">
              <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
              <geom size="0.05 0.05 0.05" type="box" contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.2 1" />
          </body>
          <body name="pedestal" pos="0 0 0">
              <inertial pos="0 0 0" quat="0.659267 -0.259505 -0.260945 0.655692" mass="60.864" diaginertia="6.0869 5.81635 4.20915" />
              <geom pos="0.26 0.345 -0.91488" quat="0.5 0.5 -0.5 -0.5" type="mesh" contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.2 1" mesh="pedestal" />
              <geom size="0.18 0.31" pos="-0.02 0 -0.29" type="cylinder" rgba="0.2 0.2 0.2 0" />
          </body>
          <body name="right_arm_base_link" pos="0 0 0">
              <inertial pos="-0.0006241 -2.8025e-05 0.065404" quat="-0.209285 0.674441 0.227335 0.670558" mass="2.0687" diaginertia="0.00740351 0.00681776 0.00672942" />
              <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="base" />
              <geom size="0.08 0.12" pos="0 0 0.12" type="cylinder" rgba="0.5 0.1 0.1 0" />
              <body name="right_l0" pos="0 0 0.08">
                  <inertial pos="0.024366 0.010969 0.14363" quat="0.894823 0.00899958 -0.170275 0.412573" mass="5.3213" diaginertia="0.0651588 0.0510944 0.0186218" />
                  <joint name="right_j0" pos="0 0 0" axis="0 0 1" limited="true" range="-3.0503 3.0503" damping="5"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l0" />
                  <body name="head" pos="0 0 0.2965">
                      <inertial pos="0.0053207 -2.6549e-05 0.1021" quat="0.999993 7.08405e-05 -0.00359857 -0.000626247" mass="1.5795" diaginertia="0.0118334 0.00827089 0.00496574" />
                      <!-- <joint name="head_pan" pos="0 0 0" axis="0 0 1" limited="true" range="-5.0952 0.9064" damping="10"/> -->
                      <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="head" />
                      <!-- <geom size="0.18" pos="0 0 0.08" rgba="0.5 0.1 0.1 0" /> -->
                      <body name="screen" pos="0.03 0 0.105" quat="0.5 0.5 0.5 0.5">
                          <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
                          <geom size="0.12 0.07 0.001" type="box" contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.2 0" />
                          <!-- <geom size="0.001" rgba="0.2 0.2 0.2 0" /> -->
                      </body>
                      <body name="head_camera" pos="0.0228027 0 0.216572" quat="0.342813 -0.618449 0.618449 -0.342813">
                          <inertial pos="0.0228027 0 0.216572" quat="0.342813 -0.618449 0.618449 -0.342813" mass="0" diaginertia="0 0 0" />
                          <site name="headsite" pos="0 0 0" size="0.01" />
                      </body>
                  </body>
                  <body name="right_torso_itb" pos="-0.055 0 0.22" quat="0.707107 0 -0.707107 0">
                      <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
                  </body>
                  <body name="right_l1" pos="0.081 0.05 0.237" quat="0.5 -0.5 0.5 0.5">
                      <inertial pos="-0.0030849 -0.026811 0.092521" quat="0.424888 0.891987 0.132364 -0.0794296" mass="4.505" diaginertia="0.0224339 0.0221624 0.0097097" />
                      <!--<joint name="right_j1" pos="0 0 0" axis="0 0 1" limited="true" range="-3.8095 2.2736" damping="10"/>-->
                      <joint name="right_j1" pos="0 0 0" axis="0 0 1"
                             limited="true" range="-3.8 -0.5"
                             damping="10"/>
                      <!--<joint name="right_j1" pos="0 0 0" axis="0 0 1" limited="true" range="0.8095 2.2736" damping="10"/>-->
                      <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l1" />
                      <!-- <geom size="0.07" pos="0 0 0.1225" rgba="0.5 0.1 0.1 0" /> -->
                      <body name="right_l2" pos="0 -0.14 0.1425" quat="0.707107 0.707107 0 0">
                          <inertial pos="-0.00016044 -0.014967 0.13582" quat="0.707831 -0.0524761 0.0516007 0.702537" mass="1.745" diaginertia="0.0257928 0.025506 0.00292515" />
                          <joint name="right_j2" pos="0 0 0" axis="0 0 1" limited="true" range="-3.0426 3.0426" damping="5"/>
                          <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l2" />
                          <geom size="0.06 0.17" pos="0 0 0.08" type="cylinder" rgba="0.5 0.1 0.1 0" />
                          <body name="right_l3" pos="0 -0.042 0.26" quat="0.707107 -0.707107 0 0">
                              <site name="armsite" pos="0 0 0" size="0.01" />
                              <inertial pos="-0.0048135 -0.0281 -0.084154" quat="0.902999 0.385391 -0.0880901 0.168247" mass="2.5097" diaginertia="0.0102404 0.0096997 0.00369622" />
                              <joint name="right_j3" pos="0 0 0" axis="0 0 1" limited="true" range="-3.0439 3.0439" damping="5"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l3" />
                              <!-- <geom size="0.06" pos="0 -0.01 -0.12" rgba="0.5 0.1 0.1 0" /> -->
                              <body name="right_l4" pos="0 -0.125 -0.1265" quat="0.707107 0.707107 0 0">
                                  <inertial pos="-0.0018844 0.0069001 0.1341" quat="0.803612 0.031257 -0.0298334 0.593582" mass="1.1136" diaginertia="0.0136549 0.0135493 0.00127353" />
                                  <joint name="right_j4" pos="0 0 0" axis="0 0 1" limited="true" range="-2.9761 2.9761" damping="5" />
                                  <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l4" />
                                  <geom size="0.045 0.15" pos="0 0 0.11" type="cylinder" rgba="0.5 0.1 0.1 0" />
                                  <body name="right_arm_itb" pos="-0.055 0 0.075" quat="0.707107 0 -0.707107 0">
                                      <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
                                  </body>
                                  <body name="right_l5" pos="0 0.031 0.275" quat="0.707107 -0.707107 0 0">
                                      <inertial pos="0.0061133 -0.023697 0.076416" quat="0.404076 0.9135 0.0473125 0.00158335" mass="1.5625" diaginertia="0.00474131 0.00422857 0.00190672" />
                                      <joint name="right_j5" pos="0 0 0" axis="0 0 1" limited="true" range="-2.9761 2.9761" damping="5"/>
                                      <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l5" />
                                      <!-- <geom size="0.06" pos="0 0 0.1" rgba="0.5 0.1 0.1 0" /> -->
                                      <body name="right_hand_camera" pos="0.039552 -0.033 0.0695" quat="0.707107 0 0.707107 0">
                                          <inertial pos="0.039552 -0.033 0.0695" quat="0.707107 0 0.707107 0" mass="0" diaginertia="0 0 0" />
                                      </body>
                                      <body name="right_wrist" pos="0 0 0.10541" quat="0.707107 0.707107 0 0">
                                          <inertial pos="0 0 0.10541" quat="0.707107 0.707107 0 0" mass="0" diaginertia="0 0 0" />
                                      </body>
                                      <body name="right_l6" pos="0 -0.11 0.1053" quat="0.0616248 0.06163 -0.704416 0.704416">
                                          <inertial pos="-8.0726e-06 0.0085838 -0.0049566" quat="0.479044 0.515636 -0.513069 0.491322" mass="0.3292" diaginertia="0.000360258 0.000311068 0.000214974" />
                                          <joint name="right_j6" pos="0 0 0" axis="0 0 1" limited="true" range="-4.7124 4.7124" damping="5"/>
                                          <geom type="mesh" contype="4" conaffinity="2" group="1" rgba="0.5 0.1 0.1 1" mesh="l6" />
                                          <geom size="0.055 0.025" pos="0 0.015 -0.01" type="cylinder" rgba="0.5 0.1 0.1 0" />
                                          <body name="right_hand" pos="0 0 0.0245" quat="0.707107 0 0 0.707107">
                                              <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                                              <geom type="mesh" contype="1" conaffinity="1" group="1" rgba="0.5 0.1 0.1 1" pos= "0 0 0.03" mesh="eGripperBase" />

                                              <geom size="0.035 0.014" pos="0 0 0.015" type="cylinder" rgba="0 0 0 1"/>
                                              <!-- <geom size="0.035 0.015" pos="0 0 0.02" type="cylinder" rgba="0.2 0.2 0.2 0"/> -->

  <!--  ================= BEGIN GRIPPER ================= /-->
                                              <!-- <body name="hand" pos="0 0 0"
                                                    quat="-1 0 1 0">
                                                  <geom class="1" name="Geomclaw" type="box" size="0.01 0.04 0.01"/>
                                                      <body name="rightclaw" pos=".03 -.03 0.0" >
                                                          <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                                                          <geom
                                                                  name="rightclaw_it" condim="4" contype="2" conaffinity="2" class="1" mass="0.08" type="box" pos="0 0 0" size="0.025 0.005 0.02" rgba="0.0 1.0 0.0 1.0" friction="1 0.05 0.01"
                                                                  euler="0 0 0.2"
                                                          />
                                                          <joint name="rc_close" type="slide" pos="0 0 0" axis="0 1 0" range="0 .04" user="008" limited="true"/>
                                                          <site name="endeffector2" pos=".015 .01 0" size="0.008" rgba="0.0 0.0 0.0 0.0" />
                                                      </body>
                                                      <body name="leftclaw" pos=".03 .03 0">
                                                          <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                                                          <geom
                                                                  name="leftclaw_it" condim="4" contype="2" conaffinity="2" class="1" type="box" mass="0.08" pos="0 0 0" size="0.025 0.005 0.02" rgba="0.0 1.0 0.0 1.0" friction="1 0.05 0.01"
                                                                  euler="0 0 -0.2"
                                                          />
                                                          <joint name="lc_close" type="slide" pos="0 0 0" axis="0 -1 0" range="-.04 0" user="008" limited="true"/>
                                                          <site name="endeffector" pos=".015 -.01 0" size="0.008" rgba="0.0 0.0 0.0 0.0"  />
                                                      </body>
                                              </body> -->
                                              <body name="hand" pos="0 0 0.12" quat="-1 0 1 0">
                                                  <camera name="behindGripper" mode="track" pos="0 0 -0.5" quat="0 1 0 0" fovy="60" />
                                                  <camera name="gripperPOV" mode="track" pos="0.04 -0.06 0" quat="-1 -1.3 0 0" fovy="90" />

                                                  <site name="endEffector" pos="0.04 0 0" size="0.01" rgba='1 1 1 0' />
                                                  <geom name="rail" type="box" pos="-0.05 0 0" density="7850" size="0.005 0.055 0.005"  rgba="0.5 0.5 0.5 1.0" condim="3" friction="2 0.1 0.002"   />

                                                  <!--IMPORTANT: For rougher contact with gripper, set higher friciton values for the other interacting objects -->
                                                  <body name="rightclaw" pos="0 -0.05 0" >

                                                      <geom class="base_col" name="rightclaw_it" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015"  rgba="1 1 1 1.0"   />

                                                      <joint name="r_close" pos="0 0 0" axis="0 1 0" range= "0 0.04" armature="100" damping="1000" limited="true"  type="slide"/>

                                                      <!-- <site name="rightEndEffector" pos="0.0 0.005 0" size="0.044 0.008 0.012" type='box' /> -->
                                                      <!-- <site name="rightEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
                                                      <site name="rightEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/>
                                                      <body name="rightpad" pos ="0 .003 0" >
                                                          <geom name="rightpad_geom" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015" rgba="1 1 1 1.0" solimp="0.95 0.99 0.01" solref="0.01 1" friction="2 0.1 0.002" contype="1" conaffinity="1" mass="1"/>
                                                      </body>

                                                  </body>

                                                  <body name="leftclaw" pos="0 0.05 0">
                                                      <geom class="base_col" name="leftclaw_it" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015"  rgba="0 1 1 1.0"  />
                                                      <joint name="l_close" pos="0 0 0" axis="0 1 0" range= "-0.03 0" armature="100" damping="1000" limited="true"  type="slide"/>
                                                      <!-- <site name="leftEndEffector" pos="0.0 -0.005 0" size="0.044 0.008 0.012" type='box' /> -->
                                                      <!-- <site name="leftEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
                                                      <site name="leftEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/>
                                                      <body name="leftpad" pos ="0 -.003 0" >
                                                          <geom name="leftpad_geom" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015" rgba="0 1 1 1.0" solimp="0.95 0.99 0.01" solref="0.01 1" friction="2 0.1 0.002"  contype="1" conaffinity="1" />
                                                      </body>

                                                  </body>
                                              </body>
  <!--  ================= END GRIPPER ================= /-->
                                          </body>
                                      </body>
                                  </body>
                                  <body name="right_l4_2" pos="0 0 0">
                                      <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                                      <!-- <geom size="0.06" pos="0 0.01 0.26"
                                            rgba="0.2 0.2 0.2 0"
                                            contype="0"
                                            conaffinity="0"
                                      /> -->
                                  </body>
                              </body>
                          </body>
                          <body name="right_l2_2" pos="0 0 0">
                              <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                              <!-- <geom size="0.06" pos="0 0 0.26" rgba="0.2 0.2 0.2 0"
                                    contype="0"
                                    conaffinity="0"
                              /> -->
                          </body>
                      </body>
                      <body name="right_l1_2" pos="0 0 0">
                          <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                          <geom size="0.07 0.07" pos="0 0 0.035" type="cylinder" rgba="0.2 0.2 0.2 0"/>
                      </body>
                  </body>
              </body>
          </body>
      </body>

      <body mocap="true" name="mocap" pos="0 0 0" quat="1 0 0 0">
          <!--For debugging, set the alpha to 1-->
          <!-- <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0.5 0.5 0.5 1" size="0.1 0.02 0.02" solimp="0.99 0.99 0.01" type="box"></geom> -->
          <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0.0 0.5 0.5 0" size="0.01" type="sphere"></geom>
          <site name="mocap" pos="0 0 0" rgba="0.0 0.5 0.5 0" size="0.01" type="sphere"></site>
      </body>

</mujocoinclude>
```
