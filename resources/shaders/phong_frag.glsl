#version 120
    
// uniform sampler2D aoTexture;
// uniform vec2 windowSize;
// uniform vec3 lightsCol[1];
uniform vec3 lightPos;
uniform vec3 ka;
uniform vec3 kd;
uniform vec3 ks;
uniform float s;

varying vec3 vPos; // vertex position in camera space
varying vec3 vNor; // vertex normal in camera space
// varying vec3 vTex; // vertex texture
    
void main()
{

    // vec3 fragColor = ka * aoFactor;
    vec3 fragColor = ka;

    // Direct Lighting
    vec3 l_hat = normalize(lightPos - vPos);
    vec3 e_hat = normalize(-vPos);
    vec3 h_hat = normalize(l_hat + e_hat);
    float r = distance(lightPos, vPos);

    float diffuse = max(0, dot(l_hat, vNor));
    float specular = max(0, pow(dot(h_hat, vNor), s));
    vec3 color = (kd * diffuse + ks * specular); // * color
    fragColor += color;
    
    // Frag Color
    gl_FragColor = vec4(fragColor, 1.0);
}