#version 120

varying vec3 vPos; // in camera space
varying vec3 vNor; // in camera space
    
uniform vec3 ke;
uniform vec3 kd;
    
void main()
{
    gl_FragData[0].xyz = vPos;
    gl_FragData[1].xyz = vNor;
    gl_FragData[2].xyz = ke;
    gl_FragData[3].xyz = kd;
}