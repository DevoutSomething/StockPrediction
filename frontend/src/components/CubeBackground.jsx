import React, { useEffect, useRef } from "react";
import * as THREE from "three";

const RotatingCube = () => {
  const mountRef = useRef(null);

  useEffect(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    mountRef.current.appendChild(renderer.domElement);

    const size = 10;
    const geometry = new THREE.BoxGeometry(size, size, size);

    const colors = [
      new THREE.Color(0xa3d8f4), // Pastel blue
      new THREE.Color(0xf4a3d8), // Pastel pink
    ];
    const vertexColors = [];
    for (let i = 0; i < 36; i++) {
      const color = colors[Math.floor(i / 6) % 2];
      vertexColors.push(color.r, color.g, color.b);
    }
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true, // Enable vertex colors
      side: THREE.DoubleSide, // Render both sides of the cube
      shininess: 10, // Reduce shininess for softer lighting
    });
    geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(vertexColors, 3)
    );

    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    cube.rotation.x = Math.PI / 4; // Rotate 45 degrees around the X-axis
    cube.rotation.y = Math.PI / 4; // Rotate 45 degrees around the Y-axis

    // Add soft lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 3); // Soft ambient light
    scene.add(ambientLight);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      cube.rotation.x += 0.005; // Very slow rotation along the X-axis

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
      ref={mountRef}
      style={{ position: "fixed", top: 0, left: 0, zIndex: -5 }}
    />
  );
};

export default RotatingCube;
