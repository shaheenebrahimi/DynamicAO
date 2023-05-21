#version 120
    
uniform sampler2D posTexture;
uniform sampler2D norTexture;
uniform sampler2D keTexture;
uniform sampler2D kdTexture;
uniform vec2 windowSize;
uniform vec3 lightsPos[50];
uniform vec3 lightsCol[50];
    
void main()
{
    vec2 tex;
    tex.x = gl_FragCoord.x/windowSize.x;
    tex.y = gl_FragCoord.y/windowSize.y;
    
    // Fetch shading data
    vec3 pos = texture2D(posTexture, tex).rgb;
    vec3 nor = texture2D(norTexture, tex).rgb;
    vec3 ke = texture2D(keTexture, tex).rgb;
    vec3 kd = texture2D(kdTexture, tex).rgb;
    
    // Calculate lighting here
    vec3 fragColor = ke;

    for (int i = 0; i < 50; ++i) {
        vec3 l_hat = normalize(lightsPos[i] - pos);
        vec3 e_hat = normalize(-pos);
        vec3 h_hat = normalize(l_hat + e_hat);
        float r = distance(lightsPos[i], pos);

        float diffuse = max(0, dot(l_hat, nor));
        float specular = max(0, pow(dot(h_hat, nor), 10.0));
        float attenuation = 2.0 / (1.0 + 0.0429*abs(r) + 0.9857*pow(r, 2.0));
        vec3 color = lightsCol[i] * (kd * diffuse + vec3(1.0,1.0,1.0) * specular);
        fragColor += color * attenuation;
    }
    
    // Deferred Rendering
    gl_FragColor = vec4(fragColor, 1.0);
    // gl_FragColor = vec4(pos, 1.0);
    // gl_FragColor = vec4(nor, 1.0);
    // gl_FragColor = vec4(ke, 1.0);
    // gl_FragColor = vec4(kd, 1.0);
}