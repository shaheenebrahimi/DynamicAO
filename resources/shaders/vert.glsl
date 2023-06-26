#version 120

uniform mat4 P;
uniform mat4 MV;
uniform mat4 itMV; // inverse transpose

attribute vec4 aPos; // in object space
attribute vec3 aNor; // in object space

varying vec3 vPos; // vertex position
varying vec3 vNor; // vertex normal

void main()
{
	vPos = vec3(MV * aPos); // to view space
	vNor = normalize(vec3(itMV * vec4(aNor, 0)));
	gl_Position = P * MV * aPos; // to clip space
}
