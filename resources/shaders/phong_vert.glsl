#version 120

uniform mat4 P;
uniform mat4 MV;
uniform mat4 inverse;
attribute vec4 aPos; // in model space
attribute vec3 aNor; // in model space
// attribute vec2 aTex;
varying vec3 vPos; // vertex position
varying vec3 vNor; // vertex normal
// varying vec3 vTex; // vertex texture

void main()
{
	gl_Position = P * MV * aPos; // to clip space
	vPos = vec3(MV * aPos); // to camera space
	vNor = normalize(vec3(inverse * vec4(aNor, 0.0))); // to camera space
	// vTex = aTex;
}
