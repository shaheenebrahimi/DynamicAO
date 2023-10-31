#version 120

uniform mat4 P;
uniform mat4 MV;
uniform mat4 itMV; // inverse transpose

attribute vec4 aPos; // in object space
attribute vec3 aNor; // in object space
attribute vec2 aTex; // in texture space
attribute float aOcc; // ambient occlusion factor

varying vec3 vPos; // vertex position
varying vec3 vNor; // vertex normal
varying vec2 vTex; // vertex texture coord
varying float vOcc; // vertex occlusion factor

void main()
{
	vPos = vec3(MV * aPos); // to view space
	vNor = normalize(vec3(itMV * vec4(aNor, 0)));
	vTex = aTex;
	vOcc = aOcc;
	gl_Position = P * MV * aPos; // to clip space
}
