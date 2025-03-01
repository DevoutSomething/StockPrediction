import React, { useEffect, useRef } from "react";
import * as THREE from "three";

const RollingHills = () => {
  const mountRef = useRef(null);

  useEffect(() => {
    // Set up the scene, camera, and renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background
    mountRef.current.appendChild(renderer.domElement);

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 1);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    // Create a wave geometry in 3D (X, Y, Z)
    const waveGeometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    const amplitude = 2; // Height of the wave
    const frequency = 0.5; // Frequency of the wave
    const segments = 150; // Number of segments in the wave
    const depth = 200; // Depth of the wave (Z direction)

    for (let i = 0; i <= segments; i++) {
      for (let j = 0; j <= depth; j++) {
        const x = (i / segments) * 20 - 10; // Spread the wave along the X-axis
        const z = (j / depth) * 20 - 10; // Spread the wave along the Z-axis
        const y = Math.sin(Math.sqrt(x * x + z * z) * frequency) * amplitude; // 3D sine wave

        vertices.push(x, y - 2, z); // Move the wave down by subtracting 2 from the y-axis

        const color = new THREE.Color();
        color.setHSL(0.5 + (i / segments) * 0.3, 1, 0.5); // Hue from 0.5 (blue) to 0.8 (green)
        colors.push(color.r, color.g, color.b);
      }
    }

    // Set the vertices and colors in the geometry
    waveGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3)
    );
    waveGeometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(colors, 3)
    );

    const waveMaterial = new THREE.LineBasicMaterial({ vertexColors: true });

    const waveMesh = new THREE.LineSegments(waveGeometry, waveMaterial);
    scene.add(waveMesh);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      const positions = waveGeometry.attributes.position.array;
      const time = Date.now() * 0.0001; // Slow down the wave movement by adjusting the multiplier
      for (let i = 0; i <= segments; i++) {
        for (let j = 0; j <= depth; j++) {
          const x = (i / segments) * 20 - 10;
          const z = (j / depth) * 20 - 10;

          positions[(i * (depth + 1) + j) * 3 + 1] =
            Math.sin(
              Math.sqrt((x + time) * (x + time) + (z + time) * (z + time)) *
                frequency
            ) * amplitude;
          positions[(i * (depth + 1) + j) * 3 + 0] =
            x + Math.sin(time * 0.1) * 0.5; // Rolling effect in X
          positions[(i * (depth + 1) + j) * 3 + 2] =
            z + Math.cos(time * 0.1) * 0.5; // Rolling effect in Z
        }
      }
      waveGeometry.attributes.position.needsUpdate = true;

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      if (mountRef.current) {
        window.removeEventListener("resize", handleResize);
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        zIndex: -1,
      }}
      ref={mountRef}
    />
  );
};

export default RollingHills;
