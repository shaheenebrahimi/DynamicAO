#version 120

varying vec3 vPos; // in camera space
varying vec3 vNor; // in camera space
    
void main()
{
    gl_FragData[0].xyz = vPos;
}