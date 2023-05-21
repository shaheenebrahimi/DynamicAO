#version 120

uniform mat4 P;
uniform mat4 MV;
uniform mat4 inverse;
uniform float t;
attribute vec4 aPos; // In object space
attribute vec3 aNor; // In object space
varying vec3 vPos; // vertex position
varying vec3 vNor; // In camera space

void main()
{
	vec3 norm = aNor;

	float theta = aPos.y;
	float x = aPos.x;

	float f = cos(x+t) + 2;
	float df = -sin(x+t);

	float y = f * cos(theta);
	float z = f * sin(theta);
	vec3 position = vec3 (x, y, z);

	vec3 dpdx = vec3 (1, df*cos(theta), df*sin(theta));
	vec3 dpdt = vec3 (0, -f*sin(theta), f*cos(theta));
	norm = cross(dpdt, dpdx);
	vec3 n_hat = normalize(norm);

    vPos = vec3(MV * vec4(position, 1.0));
	vNor = normalize(vec3(inverse * vec4(n_hat, 0.0))); // Assuming MV contains only translations and rotations
	gl_Position = P * (MV * vec4(position, 1.0));
}
