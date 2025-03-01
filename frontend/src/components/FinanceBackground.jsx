import React, { useEffect, useRef } from "react";
import * as THREE from "three";

const SwirlAnimation = () => {
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
    camera.position.z = 50; // Position the camera away from the center

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0xdcdcdc, 0); // Transparent background

    mountRef.current.appendChild(renderer.domElement);

    const particleCount = 5000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 100; // X
      positions[i * 3 + 1] = (Math.random() - 0.5) * 100; // Y
      positions[i * 3 + 2] = (Math.random() - 0.5) * 100; // Z

      if (i % 2 === 0) {
        // Pink
        colors[i * 3] = 1; // Red
        colors[i * 3 + 1] = 0.4; // Green
        colors[i * 3 + 2] = 0.7; // Blue
      } else {
        // Blue
        colors[i * 3] = 0.2; // Red
        colors[i * 3 + 1] = 0.4; // Green
        colors[i * 3 + 2] = 1; // Blue
      }
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.5, // Size of particles
      vertexColors: true, // Enable vertex colors
      blending: THREE.AdditiveBlending, // Additive blending for bright effects
      transparent: true,
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    const animate = () => {
      requestAnimationFrame(animate);

      particles.rotation.y += 0.0005; // Slower rotation around Y-axis
      particles.rotation.x += 0.00025; // Slower rotation around X-axis
      const positions = geometry.attributes.position.array;
      const time = Date.now() * 0.00005; // Slower time-based animation
      for (let i = 0; i < particleCount; i++) {
        const x = positions[i * 3];
        const y = positions[i * 3 + 1];
        const z = positions[i * 3 + 2];

        positions[i * 3] +=
          Math.sin(y * 0.003 + time) * 0.01 + (Math.random() - 0.5) * 0.005; // Slower X
        positions[i * 3 + 1] +=
          Math.cos(x * 0.003 + time) * 0.01 + (Math.random() - 0.5) * 0.005; // Slower Y
        positions[i * 3 + 2] +=
          Math.sin(z * 0.003 + time) * 0.01 + (Math.random() - 0.5) * 0.005; // Slower Z
      }
      geometry.attributes.position.needsUpdate = true; // Update position data

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      if (mountRef.current) {
        window.removeEventListener("resize", handleResize);
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={mountRef}
      style={{ position: "fixed", width: "100%", height: "100vh" }}
    />
  );
};

export default SwirlAnimation;
