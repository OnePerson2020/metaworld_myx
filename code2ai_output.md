# 项目导出

**文件数量**: 14  
**总大小**: 83.1 KB  
**Token 数量**: 22.6K  
**生成时间**: 2025/8/26 18:35:04

## 文件结构

```
📁 .
  📁 ppo_test
    📁 xml
      📄 basic_scene.xml
      📄 peg_block_dependencies.xml
      📄 peg_block.xml
      📄 peg_insert_dependencies.xml
      📄 sawyer_peg_insertion_side.xml
      📄 xyz_base_dependencies.xml
      📄 xyz_base.xml
    📄 __init__.py
    📄 reward_utils.py
    📄 sawyer_peg_insertion_side_v4.py
    📄 sawyer_xyz_env.py
    📄 types.py
  📄 evaluate_rl.py
  📄 train_rl.py
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

### ppo_test/xml/peg_block_dependencies.xml

*大小: 1.2 KB | Token: 312*

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

*大小: 2.3 KB | Token: 588*

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

            <!-- <geom type="box" conaffinity="1" contype="1" group="1" material="peg_block_wood"
                  size="0.03 0.017071 0.005"
                  pos="0 -0.101 0.175"
                  euler="-45 0 0"/>
            
            <geom type="box" conaffinity="1" contype="1" group="1" material="peg_block_wood"
                  size="0.03 0.017071 0.005"
                  pos="0 -0.101 0.085"
                  euler="45 0 0"/>

            <geom type="box" conaffinity="1" contype="1" group="1" material="peg_block_wood"
                  size="0.017071 0.005 0.03"
                  pos="-0.042 -0.11 0.13"
                  euler="0 0 45"/>

            <geom type="box" conaffinity="1" contype="1" group="1" material="peg_block_wood"
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

*大小: 1.8 KB | Token: 468*

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
          <geom name="pegHead_geom" type="box" size="0.005 0.016 0.016" pos="-0.12 0 0" mass=".001" rgba="1 0 0 0.5" conaffinity="1" contype="1" group="1"/>
          <site name="pegHead" pos="-0.12 0 0" size="0.005" rgba="0.8 0 0 0"/>
          <geom name="pegEnd_geom" type="box" size="0.005 0.016 0.016" pos="0.12 0 0" mass=".001" rgba="1 0 0 0.5" conaffinity="1" contype="1" group="1"/>
          <site name="pegEnd" pos="0.12 0 0" size="0.005" rgba="0.8 0 0 0"/>
          <site name="pegGrasp" pos=".0 .0 .0" size="0.005" rgba="0.8 0 0 1"/>
        </body>

        <body name="box" euler="0 0 1.57" pos="-0.3 0.6 0">
          <include file="./peg_block.xml"/>
          <site name="goal" pos="0.0 0.03 0.13" size="0.005" rgba="0.8 0 0 1"/>
        </body>
        

    </worldbody>

    <actuator>
      <position joint="r_close" kp="300" ctrllimited="true" ctrlrange="0 0.04"/>
      <position joint="l_close" kp="300" ctrllimited="true" ctrlrange="-0.03 0"/>
    </actuator>

    <equality>
        <weld body1="mocap" body2="hand" solref="0.001 1"></weld>
    </equality>
    
    <sensor>
        <force name="peg_force_sensor" site="pegGrasp"/>
        <torque name="peg_torque_sensor" site="pegGrasp"/>
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

*大小: 18.9 KB | Token: 5.3K*

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

                                                      <joint name="r_close" pos="0 0 0" axis="0 1 0" range="0 0.04" armature="0.1" damping="20" limited="true" type="slide"/>
                                                      <!-- <site name="rightEndEffector" pos="0.0 0.005 0" size="0.044 0.008 0.012" type='box' /> -->
                                                      <!-- <site name="rightEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
                                                      <site name="rightEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 .0"/>
                                                      <body name="rightpad" pos ="0 .003 0" >
                                                          <geom name="rightpad_geom" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015" rgba="1 1 1 1.0" solimp="0.95 0.99 0.01" solref="0.01 1" friction="2 0.1 0.002" contype="1" conaffinity="1" mass="1"/>
                                                      </body>

                                                  </body>

                                                  <body name="leftclaw" pos="0 0.05 0">
                                                      <geom class="base_col" name="leftclaw_it" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015"  rgba="0 1 1 1.0"  />
                                                      <joint name="l_close" pos="0 0 0" axis="0 1 0" range="-0.03 0" armature="0.1" damping="20" limited="true" type="slide"/>
                                                      <!-- <site name="leftEndEffector" pos="0.0 -0.005 0" size="0.044 0.008 0.012" type='box' /> -->
                                                      <!-- <site name="leftEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
                                                      <site name="leftEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 .0"/>
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

### ppo_test/__init__.py

*大小: 907 B | Token: 233*

```python
import pickle
import numpy as np
from .sawyer_peg_insertion_side_v4 import SawyerPegInsertionSideEnvV4
from .types import Task
import gymnasium as gym

def make_env(seed=None, render_mode=None, max_steps=None, print_flag=False, pos_action_scale=0.01):
    env = SawyerPegInsertionSideEnvV4(render_mode=render_mode, print_flag=print_flag, pos_action_scale=pos_action_scale)
    # 创建一个随机任务：每次 reset 时都会重新随机初始化
    task_data = {
        "env_cls": SawyerPegInsertionSideEnvV4,
        "partially_observable": False,
        "freeze": False,
        "seeded_rand_vec": True,
    }
    env.set_task(Task(env_name="peg-insert-side-v4", data=pickle.dumps(task_data)))

    if seed is not None:
        env.seed(seed)

    # 如果需要限制最大步数
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env
```

### ppo_test/reward_utils.py

*大小: 8.2 KB | Token: 2.3K*

```python
"""A set of reward utilities written by the authors of dm_control."""
from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


SIGMOID_TYPE = Literal[
    "gaussian",
    "hyperbolic",
    "long_tail",
    "reciprocal",
    "cosine",
    "linear",
    "quadratic",
    "tanh_squared",
]

X = TypeVar("X", float, npt.NDArray, np.floating)


def _sigmoids(x: X, value_at_1: float, sigmoid: SIGMOID_TYPE) -> X:
    """Maps the input to values between 0 and 1 using a specified sigmoid function. Returns 1 when the input is 0, between 0 and 1 otherwise.

    Args:
        x: The input.
        value_at_1: The output value when `x` == 1. Must be between 0 and 1.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.

    Returns:
        The input mapped to values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}."
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
            )

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
    x: X,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float | np.floating[Any] = 0.0,
    sigmoid: SIGMOID_TYPE = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> X:
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: The input.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.
        value_at_margin: A value between 0 and 1 specifying the output when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError(f"`margin` must be non-negative. Current value: {margin}")

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value.item() if np.isscalar(x) else value


def inverse_tolerance(
    x: X,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: SIGMOID_TYPE = "reciprocal",
) -> X:
    """Returns 0 when `x` falls inside the bounds, between 1 and 0 otherwise.

    Args:
        x: The input
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.
        value_at_margin: A value between 0 and 1 specifying the output when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    bound = tolerance(
        x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=0
    )
    return 1 - bound


def rect_prism_tolerance(
    curr: npt.NDArray[np.float_],
    zero: npt.NDArray[np.float_],
    one: npt.NDArray[np.float_],
) -> float:
    """Computes a reward if curr is inside a rectangular prism region.

    All inputs are 3D points with shape (3,).

    Args:
        curr: The point that the prism reward region is being applied for.
        zero: The diagonal opposite corner of the prism with reward 0.
        one: The corner of the prism with reward 1.

    Returns:
        A reward if curr is inside the prism, 1.0 otherwise.
    """

    def in_range(a, b, c):
        return float(b <= a <= c) if c >= b else float(c <= a <= b)

    in_prism = (
        in_range(curr[0], zero[0], one[0])
        and in_range(curr[1], zero[1], one[1])
        and in_range(curr[2], zero[2], one[2])
    )
    if in_prism:
        diff = one - zero
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
    else:
        return 1.0


def hamacher_product(a: float, b: float) -> float:
    """Returns the hamacher (t-norm) product of a and b.

    Computes (a * b) / ((a + b) - (a * b)).

    Args:
        a: 1st term of the hamacher product.
        b: 2nd term of the hamacher product.

    Returns:
        The hammacher product of a and b

    Raises:
        ValueError: a and b must range between 0 and 1
    """
    if not ((0.0 <= a <= 1.0) and (0.0 <= b <= 1.0)):
        raise ValueError(f"a ({a}) and b ({b}) must range between 0 and 1")
    denominator = a + b - (a * b)
    h_prod = ((a * b) / denominator) if denominator > 0 else 0

    assert 0.0 <= h_prod <= 1.0
    return h_prod
```

### ppo_test/sawyer_peg_insertion_side_v4.py

*大小: 28.7 KB | Token: 7.8K*

```python
# metaworld/sawyer_peg_insertion_side_v3.py

from __future__ import annotations

from typing import Any, Tuple, Callable, SupportsFloat

import pickle
import mujoco
import numpy as np
import numpy.typing as npt
from pathlib import Path

from scipy.spatial.transform import Rotation

from ppo_test.sawyer_xyz_env import SawyerMocapBase
from ppo_test.types import Task, XYZ, RenderMode
from ppo_test import reward_utils

from gymnasium.spaces import Box
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle


box_raw = 0
quat_box = Rotation.from_euler('xyz', [0, 0, 90+box_raw], degrees=True).as_quat()[[3,0, 1, 2]]

Len_observation: int = 21

_HAND_POS_SPACE = Box(
    
    np.array([-0.525, 0.348, -0.0525]),
    np.array([+0.525, 1.025, 0.7]),
    dtype=np.float64,
)
"""Bounds for hand position."""

_HAND_QUAT_SPACE = Box(
    np.array([-1.0, -1.0, -1.0, -1.0]),
    np.array([1.0, 1.0, 1.0, 1.0]),
    dtype=np.float64,
)
"""Bounds for hand quaternion."""

TARGET_RADIUS: float = 0.05
"""Upper bound for distance from the target when checking for task completion."""

class _Decorators:
    @classmethod
    def assert_task_is_set(cls, func: Callable) -> Callable:
        """Asserts that the task has been set in the environment before proceeding with the function call.
        To be used as a decorator for SawyerPegInsertionSideEnvV4 methods."""

        def inner(*args, **kwargs) -> Any:
            env = args[0]
            if not env._set_task_called:
                raise RuntimeError(
                    "You must call env.set_task before using env." + func.__name__
                )
            return func(*args, **kwargs)

        return inner
        
class SawyerPegInsertionSideEnvV4(SawyerMocapBase, EzPickle):
    
    max_path_length: int = 300
    """The maximum path length for the environment (the task horizon)."""

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        frame_skip: int = 5,
        mocap_low: XYZ | None = None,
        mocap_high: XYZ | None = None,
        print_flag: bool = False,
        pos_action_scale = 0.01,
    ) -> None:

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.5, 0.02)
        obj_high = (0.2, 0.7, 0.02)
        goal_low = (-0.35, 0.4, -0.001)
        goal_high = (-0.25, 0.7, 0.001)

        self.hand_init_pos = np.array([0, 0.6, 0.2])
        self.hand_init_quat = Rotation.from_euler('xyz', [0,90,0], degrees=True).as_quat()[[3, 0, 1, 2]]
        
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low) + Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.03, 0.0, 0.13])),
            np.array(goal_high) + Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.03, 0.0, 0.13])),
            dtype=np.float64,
        )
    
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack((mocap_low, [-1.0, -1.0, -1.0]))
        self.mocap_high = np.hstack((mocap_high, [1.0, 1.0, 1.0]))
        
        self.curr_path_length: int = 0
        self.seeded_rand_vec: bool = False
        self._freeze_rand_vec: bool = True
        self._last_rand_vec: npt.NDArray[Any] | None = None
        self.obj_init_pos: npt.NDArray[Any] | None = None

        self._partially_observable: bool = False

        self._set_task_called: bool = False
        self.print_flag = print_flag
        self.pos_action_scale = pos_action_scale

        super().__init__(
            self.model_name,
            frame_skip=frame_skip,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            width=width,
            height=height,
        )

        mujoco.mj_forward(
            self.model, self.data
        )  # *** DO NOT REMOVE: EZPICKLE WON'T WORK *** #

        self.action_space = Box(  # type: ignore
            np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            np.array([+1.0, +1.0, +1.0, +1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._prev_obs = np.zeros(Len_observation, dtype=np.float64)

        # 任务阶段初始化
        self.task_phase = 'approach'
        
        EzPickle.__init__(
            self,
            self.model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            render_mode,
            camera_id,
            camera_name,
            width,
            height,
        )
        
    @property
    def model_name(self) -> str:
        _CURRENT_FILE_DIR = Path(__file__).parent.absolute()
        ENV_ASSET_DIR_V3 = _CURRENT_FILE_DIR / 'xml'
        file_name = "sawyer_peg_insertion_side.xml"
        return str(ENV_ASSET_DIR_V3 / file_name)

    @property
    def sawyer_observation_space(self) -> Box:
        obj_low = np.full(7, -np.inf, dtype=np.float64)
        obj_high = np.full(7, +np.inf, dtype=np.float64)
        
        if self._partially_observable:
            goal_low = np.zeros(3)
            goal_high = np.zeros(3)
        else:
            assert (
                self.goal_space is not None
            ), "The goal space must be defined to use full observability"
            goal_low = self.goal_space.low
            goal_high = self.goal_space.high
            
        gripper_low = -1.0
        gripper_high = +1.0
        
        force_low = np.full(3, -20, dtype=np.float64)
        force_high = np.full(3, 20, dtype=np.float64)

        torque_low = np.full(3, -20, dtype=np.float64)
        torque_high = np.full(3, 20, dtype=np.float64)

        # return Box(
        #     np.hstack(
        #         (
        #             _HAND_POS_SPACE.low, 
        #             gripper_low,
        #             obj_low,
        #             _HAND_POS_SPACE.low, 
        #             gripper_low,
        #             obj_low,
        #             goal_low,
        #         )
        #     ),
        #     np.hstack(
        #         (
        #             _HAND_POS_SPACE.high,
        #             gripper_high,
        #             obj_high,
        #             _HAND_POS_SPACE.high,
        #             gripper_high,
        #             obj_high,
        #             goal_high,
        #         )
        #     ),
        #     dtype=np.float64,
        # )
            
        # Current observation: pos_hand (3) + quat_hand (4) + gripper_distance_apart (1) + pegHead_force (3) + pegHead_torque + obs_obj_padded (7) = 21
        # Goal: 3
        return Box(
            np.hstack(
                (
                    # Current obs (21)
                    _HAND_POS_SPACE.low, 
                    _HAND_QUAT_SPACE.low,
                    gripper_low,
                    force_low,
                    torque_low,
                    obj_low,
                    # Previous obs (21)
                    _HAND_POS_SPACE.low, 
                    _HAND_QUAT_SPACE.low,
                    gripper_low,
                    force_low,
                    torque_low,
                    obj_low,
                    # Goal (3)
                    goal_low,
                )
            ),
            np.hstack(
                (
                    # Current obs (21)
                    _HAND_POS_SPACE.high,
                    _HAND_QUAT_SPACE.high,
                    gripper_high,
                    force_high,
                    torque_high,
                    obj_high,
                    # Previous obs (21)
                    _HAND_POS_SPACE.high,
                    _HAND_QUAT_SPACE.high,
                    gripper_high,
                    force_high,
                    torque_high,
                    obj_high,
                    # Goal (3)
                    goal_high,
                )
            ),
            dtype=np.float64,
        )
        
    def seed(self, seed: int) -> list[int]:
        """Seeds the environment.

        Args:
            seed: The seed to use.

        Returns:
            The seed used inside a 1 element list.
        """
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        assert self.goal_space
        self.goal_space.seed(seed)
        return [seed]

    def set_task(self, task: Task) -> None:
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data["env_cls"])
        del data["env_cls"]
        self._freeze_rand_vec = data.get("freeze", False)
        self.seeded_rand_vec = data.get("seeded_rand_vec", True)
        self._last_rand_vec = data.get("rand_vec", None)
        self._partially_observable =  data["partially_observable"]
        del data["partially_observable"]

    def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
        """Adjusts the position of the mocap body from the given action.
        Moves each body axis in XYZ by the amount described by the action.

        Args:
            action: The action to apply (in offsets between :math:`[-1, 1]` for each axis in XYZ).
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        pos_delta = self.pos_action_scale * action[:3]
        
        # 应用位移并裁剪到工作空间
        new_mocap_pos = self.data.mocap_pos.copy()
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :] + pos_delta,
            self.mocap_low[:3],
            self.mocap_high[:3],
        )
        self.data.mocap_pos = new_mocap_pos
        
        # r_increment = Rotation.from_quat(action[3:7])
        
        # current_mocap = self.data.mocap_quat[0]
        # new_mocap_r = r_increment * Rotation.from_quat(current_mocap[[1,2,3,0]])
        # new_mocap_quat = new_mocap_r.as_quat()[[3,0,1,2]]
        # self.data.mocap_quat[0] = new_mocap_quat

    def get_peghead_force_and_torque(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        计算并返回 pegHead_geom 在世界坐标系下受到的总接触力和相对于 pegGrasp 点的总力矩。

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
            一个元组，包含：
            - total_world_force (3,): 世界坐标系下的总受力向量。
            - total_world_torque (3,): 世界坐标系下，相对于 pegGrasp 点的总力矩向量。
        """
        # --- 初始化 ---
        peg_head_geom_id = self.data.geom("pegHead_geom").id
        total_world_force = np.zeros(3)
        total_world_torque = np.zeros(3)

        grasp_point_world = self.data.body("peg").xpos
        
        # --- 遍历所有接触点 ---
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 检查当前接触是否涉及 pegHead_geom
            if contact.geom1 == peg_head_geom_id or contact.geom2 == peg_head_geom_id:
                
                # 步骤 1: 获取在“接触坐标系”下的6D力/力矩向量
                force_contact_frame = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, force_contact_frame)
                
                # 步骤 2: 将接触力从“接触坐标系”旋转到“世界坐标系”
                contact_frame_rot = contact.frame.reshape(3, 3)
                force_world_frame = contact_frame_rot @ force_contact_frame[:3]
                
                # 步骤 3: 根据牛顿第三定律确定力的正确方向
                if contact.geom1 == peg_head_geom_id:
                    # 如果 geom1 是我们的传感器，力是由它施加的，我们需要反向的力
                    force_on_peghead = -force_world_frame
                else: # contact.geom2 == peg_head_geom_id
                    # 如果 geom2 是我们的传感器，力是施加于它的，方向正确
                    force_on_peghead = force_world_frame
                
                # --- 力矩计算 ---
                # 步骤 4: 获取接触点在世界坐标系下的位置
                contact_position_world = contact.pos
                
                # 步骤 5: 计算从抓取点到接触点的矢量（力臂）
                lever_arm = contact_position_world - grasp_point_world
                
                # 步骤 6: 计算该接触力产生的力矩 (tau = r x F)
                torque_i = np.cross(lever_arm, force_on_peghead)
                
                # --- 累加总力和总力矩 ---
                total_world_force += force_on_peghead
                total_world_torque += torque_i
        
        return total_world_force, total_world_torque

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("pegGrasp")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.site("pegGrasp").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    @_Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64]
    ) -> tuple[float, dict[str, Any]]:
                                
        reward, stage_rewards = self.compute_reward_test(obs)
        insertion_info = self.get_insertion_info()
        
        success = float(insertion_info["insertion_depth"] >= 0.04)
                    
        info = {
            "success": success,
            "stage_rewards": stage_rewards,
            "unscaled_reward": reward,
        }

        return reward, info

    def compute_reward_test(
        self,
        obs: npt.NDArray[Any],
    ) -> tuple[float, dict[str, float]]:
        """
        改进版分阶段奖励函数
        阶段1: 接近 peg
        阶段2: 抓取 + 抬起
        阶段3: 对准 hole
        阶段4: 插入
        """
        tcp = obs[:3]  # 手爪位置
        tcp_opened = obs[7]
        obj = obs[14:17]  # peg 位置

        # 初始化任务阶段状态
        if not hasattr(self, 'task_phase'):
            self.task_phase = 'approach'

        # ---------------------------------
        # 阶段1: 接近 Peg
        # ---------------------------------
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        approach_margin = float(np.linalg.norm(self.obj_init_pos - self.init_tcp))
        approach_reward = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0.0, 0.0),
            margin=approach_margin,
            sigmoid="long_tail"
        )

        close_reward = 0.5 * reward_utils.tolerance(
                                tcp_opened,
                                bounds=(0.0, 0.35),
                                margin=0.65,
                                sigmoid="long_tail"
                            )

        if tcp_to_obj < 0.025:
            if self.task_phase == 'approach':
                self.task_phase = 'grasp'
            close_reward = close_reward * 2
            
            if self.task_phase == 'grasp':
                if self.touching_main_object and tcp_to_obj < 0.02 and tcp_opened < 0.4: 
                        self.task_phase = 'alignment'

        grasp_reward = reward_utils.hamacher_product(approach_reward, close_reward)
        
        # ---------------------------------
        # 阶段3: 对准 Hole
        # ---------------------------------
        
        # 获取插入信息
        insertion_info = self.get_insertion_info()
        head_to_hole = insertion_info["peg_head_pos"] - insertion_info["hole_pos"]
        hole_orientation = insertion_info["hole_orientation"]
        insertion_depth = insertion_info["insertion_depth"]

        if self.task_phase == 'alignment':
            head_to_hole_init = self.peg_head_pos_init - self._goal_pos
            lateral_offset_init = head_to_hole_init - np.dot(head_to_hole_init, hole_orientation) * hole_orientation
            lateral_offset = head_to_hole - np.dot(head_to_hole, hole_orientation) * hole_orientation

            lateral_distance = float(np.linalg.norm(lateral_offset))
            longitudinal_distance = float(np.dot(head_to_hole, hole_orientation))

            lateral_alignment_margin = float(np.linalg.norm(lateral_offset_init))
            lateral_alignment = reward_utils.tolerance(
                lateral_distance,
                bounds=(0.0, 0.02), 
                margin=lateral_alignment_margin,
                sigmoid="long_tail",
            )

            longitudinal_alignment_margin = abs(float(np.dot(head_to_hole_init, hole_orientation)))
            longitudinal_alignment = reward_utils.tolerance(
                longitudinal_distance,
                bounds=(-0.15, 0.10),  # 纵向误差 -5cm ~ +10cm
                margin=longitudinal_alignment_margin,
                sigmoid="long_tail",
            )
            alignment_reward = reward_utils.hamacher_product(lateral_alignment, longitudinal_alignment)

            if alignment_reward > 0.9:
                self.task_phase = 'insertion'
        elif self.task_phase == 'insertion':
            alignment_reward = 1.0
            
        else:
            alignment_reward = 0.0

        # ---------------------------------
        # 阶段4: 插入（平滑增长，去掉一次性 +10）
        # ---------------------------------
        insertion_reward = 0.0
        if self.task_phase == 'insertion':
            insertion_reward = insertion_depth / 0.04
            # insertion_reward = reward_utils.tolerance(
            #     insertion_depth,
            #     bounds=(0.05, 0.10),   # 5~10cm 逐步递增
            #     margin=0.08,
            #     sigmoid="long_tail",
            # )
            if lateral_distance < 0.01:  # 插入时对准额外奖励（轻微）
                insertion_reward = min(1.0, insertion_reward + 0.2 * (1.0 - lateral_distance / 0.01))

        # ---------------------------------
        # 分阶段权重（更均衡，让“抓取+抬起”占更大比重）
        # ---------------------------------
        stage_weights = {
            "approach": 1,
            "grasp": 1,
            "alignment": 1,
            "insertion": 1,
        }

        total_reward = (
            stage_weights["approach"] * approach_reward +
            stage_weights["grasp"] * grasp_reward +
            stage_weights["alignment"] * alignment_reward +
            stage_weights["insertion"] * insertion_reward
        )

        # ---------------------------------
        # 调试信息
        # ---------------------------------
        stage_rewards = {
            "approach": float(approach_reward),
            "grasp": float(grasp_reward),
            "alignment": float(alignment_reward),
            "insertion": float(insertion_reward),
            "insertion_depth": float(insertion_depth),
            "task_phase": self.task_phase,
        }

        if self.print_flag:
            values = [approach_reward,
                    grasp_reward,
                    alignment_reward,
                    insertion_reward,
                    insertion_depth]
            print(" ".join(f"{v:6.3f}" for v in values))
            print(self.task_phase)
            
        return float(total_reward), stage_rewards

    def get_hole_info(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """获取hole的位置和朝向信息。
        朝向由从 'hole' 站点指向 'goal' 站点的单位向量表示。
        """
        hole_pos = self._get_site_pos("hole")
        goal_pos = self._get_site_pos("goal")

        # 计算从 hole 指向 goal 的向量
        orientation_vec = goal_pos - hole_pos
        
        hole_orientation = orientation_vec / np.linalg.norm(orientation_vec)
            
        return goal_pos, hole_orientation

    def get_insertion_info(self) -> dict[str, Any]:
        """获取插入相关信息"""
        hole_pos, hole_orientation = self.get_hole_info() 
        peg_head_pos = self._get_site_pos("pegHead")
        
        # 计算插入深度：将 (peg头 - hole口) 的向量，投影到 hole 的朝向向量上
        insertion_depth = np.dot(peg_head_pos - hole_pos, hole_orientation)
        return {
            "hole_pos": hole_pos,
            "hole_orientation": hole_orientation,
            "peg_head_pos": peg_head_pos,
            "insertion_depth": max(0, insertion_depth),
        }

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the object.

        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy() # 前 9 个分别对应 7 个关节角度和 2 个夹爪的控制量
        qvel[9:15] = 0  # 一个刚体在空间中有 6 个自由度的速度：3 个线速度（dx, dy, dz）和 3 个角速度（ωx, ωy, ωz）。
        self.set_state(qpos, qvel)

    def _get_site_pos(self, site_name: str) -> npt.NDArray[np.float64]:
        """Gets the position of a given site.

        Args:
            site_name: The name of the site to get the position of.

        Returns:
            Flat, 3 element array indicating site's location.
        """
        return self.data.site(site_name).xpos.copy()

    @property
    def touching_main_object(self) -> bool:
        """Calls `touching_object` for the ID of the env's main object.

        Returns:
            Whether the gripper is touching the object
        """
        return self.touching_object(self.data.geom("peg").id)

    def touching_object(self, object_geom_id: int) -> bool:
        """Determines whether the gripper is touching the object with given id.

        Args:
            object_geom_id: the ID of the object in question

        Returns:
            Whether the gripper is touching the object
        """

        leftpad_geom_id = self.data.geom("leftpad_geom").id
        rightpad_geom_id = self.data.geom("rightpad_geom").id

        leftpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        rightpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in leftpad_object_contacts
        )

        rightpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in rightpad_object_contacts
        )

        return 1 < leftpad_object_contact_force and 1 < rightpad_object_contact_force

    def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:

        pos_hand = self.tcp_center
        quat_hand = self.get_endeff_quat()

        finger_right, finger_left = (
            self.data.body("rightclaw"),
            self.data.body("leftclaw"),
        )
        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obj_pos = self._get_pos_objects()
        obj_quat = self._get_quat_objects()
        pegHead_force, pegHead_torque = self.get_peghead_force_and_torque()
        pegHead_force = np.tanh(pegHead_force / 10.0)
        pegHead_torque = np.tanh(pegHead_torque / 1.0)
        
        return np.hstack((pos_hand, quat_hand, gripper_distance_apart, pegHead_force, pegHead_torque, obj_pos, obj_quat))

    def _get_obs(self) -> npt.NDArray[np.float64]:
        pos_goal = self._goal_pos
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs
        
    @_Decorators.assert_task_is_set
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment."""

        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")
            
        # 保存当前动作，用于奖励计算
        current_action = action.copy()

        # 位姿控制（位置 3 维 + 夹爪 1 维）
        self.set_xyz_action(action[:-1])
        
        u = float(np.clip(action[-1], -1, 1))
        r_ctrl = 0.02 + 0.02 * u        # 映射到 [0, 0.04]
        l_ctrl = -0.015 - 0.015 * u     # 映射到 [-0.03, 0]
        self.do_simulation([r_ctrl, l_ctrl], n_frames=self.frame_skip)

        self.curr_path_length += 1

        # 观测裁剪
        self._last_stable_obs = self._get_obs()
        self._last_stable_obs = np.clip(
            self._last_stable_obs,
            a_max=self.sawyer_observation_space.high,
            a_min=self.sawyer_observation_space.low
        ).astype(np.float64)

        # 奖励与信息
        reward, info = self.evaluate_state(self._last_stable_obs)

        # 成功判定
        terminated = False
        if info.get("stage_rewards", {}).get("insertion_depth", 0) >= 0.04:
            terminated = True
            info["success"] = 1.0
        else:
            info["success"] = 0.0

        truncated = (self.curr_path_length == self.max_path_length)

        return (
            np.array(self._last_stable_obs, dtype=np.float64),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
            pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        self.obj_init_pos = pos_peg
        self._set_obj_xyz(self.obj_init_pos)
        self.peg_head_pos_init = self._get_site_pos("pegHead")
        self.model.body("box").pos = pos_box
        self.model.body("box").quat = quat_box
        self._goal_pos = Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.0, 0.03, 0.13]))
        self.model.site("goal").pos = self._goal_pos

        return self._get_obs()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: The seed to use. Ignored, use `seed()` instead.
            options: Additional options to pass to the environment. Ignored.

        Returns:
            The `(obs, info)` tuple.
        """
        self.curr_path_length = 0
        _, info = super().reset(seed=seed, options=options)
        initial_obs_curr = self._get_curr_obs_combined_no_goal()
        self._prev_obs = initial_obs_curr.copy() 
        pos_goal = self._goal_pos
        obs = np.hstack((initial_obs_curr, self._prev_obs, pos_goal))
        return obs, info

    def _reset_hand(self, steps: int = 50) -> None:
        """Resets the hand position.

        Args:
            steps: The number of steps to take to reset the hand.
        """
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            self.data.mocap_pos[mocap_id][:] = self.hand_init_pos
            self.data.mocap_quat[mocap_id][:] = self.hand_init_quat
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

    def _get_state_rand_vec(self) -> npt.NDArray[np.float64]:
        """Gets or generates a random vector for the hand position at reset."""
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        elif self.seeded_rand_vec:
            assert self._random_reset_space is not None
            rand_vec = self.np_random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            )
            self._last_rand_vec = rand_vec
            return rand_vec
        else:
            assert self._random_reset_space is not None
            rand_vec: npt.NDArray[np.float64] = np.random.uniform(  # type: ignore
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            ).astype(np.float64)
            self._last_rand_vec = rand_vec
            return rand_vec
```

### ppo_test/sawyer_xyz_env.py

*大小: 4.2 KB | Token: 1.2K*

```python
# metaworld/sawyer_xyz_env.py

from __future__ import annotations

import copy
from functools import cached_property
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv as mjenv_gym
from gymnasium.spaces import Space

from ppo_test.types import EnvironmentStateDict, RenderMode


class SawyerMocapBase(mjenv_gym):
    """Provides some commonly-shared functions for Sawyer Mujoco envs that use mocap for XYZ control."""

    @cached_property
    def sawyer_observation_space(self) -> Space:
        raise NotImplementedError

    def __init__(
        self,
        model_name: str,
        frame_skip: int = 5,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        width: int = 480,
        height: int = 480,
    ) -> None:
        mjenv_gym.__init__(
            self,
            model_name,
            frame_skip=frame_skip,
            observation_space=self.sawyer_observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            width=width,
            height=height,
        )
        self.reset_mocap_welds()
        self.frame_skip = frame_skip

    def get_endeff_pos(self) -> npt.NDArray[Any]:
        """Returns the position of the end effector."""
        return self.data.body("hand").xpos

    def get_endeff_quat(self) -> npt.NDArray[Any]:
        """Returns the quaternion of the end effector."""
        return self.data.body("hand").xquat
    

    @property
    def tcp_center(self) -> npt.NDArray[Any]:
        """The COM of the gripper's 2 fingers.

        Returns:
            3-element position.
        """
        right_finger_pos = self.data.site("rightEndEffector")
        left_finger_pos = self.data.site("leftEndEffector")
        tcp_center = (right_finger_pos.xpos + left_finger_pos.xpos) / 2.0
        return tcp_center

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def get_env_state(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get the environment state.

        Returns:
            A tuple of (qpos, qvel).
        """
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        return copy.deepcopy((qpos, qvel))

    def set_env_state(
        self, state: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ) -> None:
        """
        Set the environment state.

        Args:
            state: A tuple of (qpos, qvel).
        """
        qpos, qvel = state
        self.set_state(qpos, qvel)

    def __getstate__(self) -> EnvironmentStateDict:
        """Returns the full state of the environment as a dict.

        Returns:
            A dictionary containing the env state from the `__dict__` method, the model name (path) and the mocap state `(qpos, qvel)`.
        """
        state = self.__dict__.copy()
        return {"state": state, "mjb": self.model_name, "mocap": self.get_env_state()}

    def __setstate__(self, state: EnvironmentStateDict) -> None:
        """Sets the state of the environment from a dict exported through `__getstate__()`.

        Args:
            state: A dictionary containing the env state from the `__dict__` method, the model name (path) and the mocap state `(qpos, qvel)`.
        """
        self.__dict__ = state["state"]
        mjenv_gym.__init__(
            self,
            state["mjb"],
            frame_skip=self.frame_skip,
            observation_space=self.sawyer_observation_space,
            render_mode=self.render_mode,
            camera_name=self.camera_name,
            camera_id=self.camera_id,
            width=self.width,
            height=self.height,
        )
        self.set_env_state(state["mocap"])

    def reset_mocap_welds(self) -> None:
        """Resets the mocap welds that we use for actuation."""
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i] = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                    )
```

### ppo_test/types.py

*大小: 533 B | Token: 147*

```python
from __future__ import annotations

from typing import Any, NamedTuple, Tuple, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias, TypedDict


class Task(NamedTuple):
    env_name: str
    data: bytes

RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"

XYZ: TypeAlias = "Tuple[float, float, float]"
"""A 3D coordinate."""

class EnvironmentStateDict(TypedDict):
    state: dict[str, Any]
    mjb: str
    mocap: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
```

### evaluate_rl.py

*大小: 5.0 KB | Token: 1.2K*

```python
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import ppo_test

# --- 1. 设置命令行参数解析 ---
parser = argparse.ArgumentParser(description="Evaluate a PPO model with an option to enable plotting.")
parser.add_argument('--plot', action='store_true', help='Enable real-time plotting of actions and observations.')
args = parser.parse_args()

# --- 2. 根据参数决定是否初始化绘图 ---
if args.plot:
    print("绘图功能已开启。")
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    # 动作和观测历史记录
    history_steps = []
    history_actions = []
    history_obs = []
    history_rewards = []
else:
    print("绘图功能已关闭。如需开启，请使用 --plot 参数运行。")

# --- 3. 加载训练好的模型 ---
model_path = "models/best_model.zip"
stats_path = "models/vec_normalize.pkl"
try:
    model = PPO.load(model_path)
    print(f"从 {model_path} 加载模型成功！")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 {model_path}，请先运行 train_rl.py。")
    exit()

# --- 4. 创建评估环境 ---
MAX_STEPS = 200
eval_env_raw = ppo_test.make_env(
    seed=999,
    render_mode="human",
    max_steps=MAX_STEPS,
    print_flag=True
)

# 使用加载的统计数据封装环境
try:
    eval_env_vec = DummyVecEnv([lambda: eval_env_raw])

    # 现在传入向量化后的环境
    env = VecNormalize.load(stats_path, eval_env_vec)
    print(f"从 {stats_path} 加载归一化统计数据成功！")
    
    env.training = False
    env.norm_reward = False
except FileNotFoundError:
    print(f"错误: 找不到归一化文件 {stats_path}，请先完整运行 train_rl.py。")
    exit()

# --- 5. 运行评估 ---
num_episodes = 10
success_count = 0

for ep in range(num_episodes):
    print(f"\n=== 开始评估第 {ep+1}/{num_episodes} 局 ===")
    obs = env.reset()

    # 如果开启绘图，则重置数据
    if args.plot:
        history_steps.clear()
        history_actions.clear()
        history_obs.clear()
        history_rewards.clear()

    # 调整摄像机
    mujoco_env = env.envs[0] 
    if hasattr(mujoco_env, 'mujoco_renderer') and mujoco_env.mujoco_renderer.viewer:
        mujoco_env.mujoco_renderer.viewer.cam.azimuth = 245
        mujoco_env.mujoco_renderer.viewer.cam.elevation = -20

    episode_reward = 0
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step}, Action: {action}")

        obs, reward, done, info = env.step(action)
        original_obs = env.get_original_obs()
        episode_reward += reward[0] 
        info_dict = info[0]

        # --- 仅在开启绘图时记录数据和更新图表 ---
        if args.plot:
            # 记录数据
            history_steps.append(step)
            history_actions.append(action.copy())
            history_obs.append(original_obs.copy())
            history_rewards.append(episode_reward)

            # 实时绘图更新
            if step % 2 == 0:  # 每2步更新一次，避免太卡
                ax1.clear()
                ax2.clear()

                # 转为 NumPy 数组便于处理
                arr_actions = np.array(history_actions)
                arr_obs = np.array(history_obs)

                # 绘制动作
                for i in range(arr_actions.shape[1]):
                    ax1.plot(history_steps, arr_actions[:, i], label=f'Action {i}')
                ax1.set_title(f"Episode {ep+1} - Actions Over Time")
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Action Value")
                ax1.legend()
                ax1.grid(True)

                # 绘制观测（取前几维或关键维度）
                obs_dim = arr_obs.shape[1]
                plot_dims = min(6, obs_dim)  # 最多画前6维
                for i in range(plot_dims):
                    ax2.plot(history_steps, arr_obs[:, i], label=f'Obs {i}')
                ax2.set_title("Observations Over Time")
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Obs Value")
                ax2.legend()
                ax2.grid(True)

                # 刷新图像
                fig.tight_layout()
                plt.pause(0.01)  # 短暂暂停以刷新

        env.render()

        # 检查成功
        if info_dict.get("success", 0.0) > 0.5:
            print("✅ 成功插入 Peg！")
            success_count += 1
            time.sleep(1.5)
            break

        step += 1

    print(f"该局累计奖励: {episode_reward:.2f}")

    # 如果绘图，则暂停一下保持图表
    if args.plot:
        time.sleep(1)

print("\n=== 评估结束 ===")
print(f"总回合数: {num_episodes}")
print(f"成功次数: {success_count}")
print(f"成功率: {success_count / num_episodes * 100:.1f}%")

# --- 6. 关闭环境和图形 ---
env.close()
if args.plot:
    plt.ioff()
    plt.show()  # 保持最终图像显示
```

### train_rl.py

*大小: 5.8 KB | Token: 1.6K*

```python
import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import ppo_test
from tqdm import tqdm


import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class TensorboardPhaseCallback(BaseCallback):
    """
    自定义回调，用于记录环境中的阶段信息到TensorBoard
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.phase_counts = {
            'approach': 0,
            'grasp': 0,
            'alignment': 0,
            'insertion': 0,
            'success': 0
        }
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.episode_phases = []

    def _on_step(self) -> bool:
        # 获取环境信息
        infos = self.locals.get('infos', [])
        
        for i, info in enumerate(infos):
            if info is None:
                continue
                
            # 记录阶段信息
            if 'stage_rewards' in info and 'task_phase' in info['stage_rewards']:
                phase = info['stage_rewards']['task_phase']
                self.phase_counts[phase] += 1
                
                # 记录阶段转换
                if hasattr(self, 'last_phase') and self.last_phase != phase:
                    self.logger.record(f'phase_transitions/{self.last_phase}_to_{phase}', 1)
            
            # 记录阶段奖励
            if 'stage_rewards' in info:
                for stage_name, stage_reward in info['stage_rewards'].items():
                    if stage_name != 'task_phase':  # 排除阶段名称本身
                        self.logger.record(f'stage_rewards/{stage_name}', stage_reward)
            
            # 记录插入深度
            if 'stage_rewards' in info and 'insertion_depth' in info['stage_rewards']:
                insertion_depth = info['stage_rewards']['insertion_depth']
                self.logger.record('metrics/insertion_depth', insertion_depth)
            
            # 记录成功次数
            if 'success' in info:
                self.phase_counts['success'] += int(info['success'])
                self.logger.record('metrics/success_rate', info['success'])
        
        # 每100步记录一次阶段统计
        if self.n_calls % 100 == 0:
            total_steps = sum(self.phase_counts.values())
            for phase, count in self.phase_counts.items():
                if total_steps > 0:
                    phase_ratio = count / total_steps
                    self.logger.record(f'phase_stats/{phase}_ratio', phase_ratio)
                    self.logger.record(f'phase_stats/{phase}_count', count)
            
            # 重置计数
            self.phase_counts = {k: 0 for k in self.phase_counts.keys()}
        
        return True

    def _on_rollout_end(self) -> bool:
        # 在rollout结束时记录更多统计信息
        infos = self.locals.get('infos', [])
        if infos:
            # 计算平均奖励
            rewards = [info.get('unscaled_reward', 0) for info in infos if info]
            if rewards:
                avg_reward = np.mean(rewards)
                self.logger.record('metrics/avg_episode_reward', avg_reward)
        
        return True
    
def make_env_fn(seed):
    def _init():
        env = ppo_test.make_env(seed=seed)
        return env
    return _init


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        self.pbar.close()


def main():
    LOG_DIR = "logs"
    MODEL_DIR = "models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Create and wrap the training environment
    num_envs = 16
    eval_frequency_adjusted = max(1, 5000 // num_envs)
    train_env = SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # 2. Create and wrap the evaluation environment in the same way
    eval_env_raw = ppo_test.make_env(seed=999)
    eval_env = DummyVecEnv([lambda: eval_env_raw])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=eval_frequency_adjusted,
        n_eval_episodes=5,
        deterministic=True,
    )
    progress_callback = ProgressBarCallback(total_timesteps=1_000_000)

    callback = CallbackList([eval_callback, progress_callback])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
        n_steps=2048,
        batch_size=128,
        ent_coef=0.01,
    )
    print(f"Starting training with n_steps = {model.n_steps}")

    model.learn(total_timesteps=1_000_000, callback=callback)

    # Don't forget to save the normalization stats for later evaluation
    stats_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    train_env.save(stats_path)
    print(f"Saved VecNormalize stats to {stats_path}")
    model.save(os.path.join(MODEL_DIR, "ppo_peg_insert_final"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()
```
