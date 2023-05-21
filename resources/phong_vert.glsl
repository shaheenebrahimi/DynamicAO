#version 120

uniform mat4 P;
uniform mat4 MV;
uniform mat4 inverse;
attribute vec4 aPos; // in object space
attribute vec3 aNor; // in object space
varying vec3 vPos; // vertex position
varying vec3 vNor; // vertex normal

void main()
{
	gl_Position = P * MV * aPos;
	vPos = vec3(MV * aPos);
	vNor = normalize(vec3(inverse * vec4(aNor, 0.0)));
}
