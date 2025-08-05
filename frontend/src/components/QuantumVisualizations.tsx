import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere } from '@react-three/drei';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

// Type definitions for quantum state data
interface ComplexNumber {
    real: number;
    imag: number;
}

interface QuantumState {
    amplitude: [ComplexNumber, ComplexNumber];
    label: string;
}

interface QuantumGate {
    type: string;
    qubit: number;
    control?: number;
    target?: number;
    params?: any;
}

interface QuantumCircuit {
    qubits: number[];
}

interface MarketDataPoint {
    value: number;
    timestamp: string;
}

// Helper function to calculate magnitude of complex number
const magnitude = (complex: ComplexNumber): number => {
    return Math.sqrt(complex.real * complex.real + complex.imag * complex.imag);
};

// Helper function to calculate phase of complex number
const phase = (complex: ComplexNumber): number => {
    return Math.atan2(complex.imag, complex.real);
};

// Helper function to format complex number for display
const formatComplex = (complex: ComplexNumber): string => {
    const mag = magnitude(complex);
    const ph = phase(complex);
    return `${mag.toFixed(2)}e^(i${ph.toFixed(2)})`;
};

// Bloch Sphere Component for Quantum State Visualization
const BlochSphere = ({ position, state, label, connections }: {
    position: THREE.Vector3;
    state: [ComplexNumber, ComplexNumber];
    label: string;
    connections: number[];
}) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const [hovered, setHovered] = useState(false);

    // Calculate Bloch vector from quantum state
    const blochVector = useMemo(() => {
        const prob0 = magnitude(state[0]) ** 2;
        const prob1 = magnitude(state[1]) ** 2;
        const theta = Math.acos(2 * prob0 - 1);
        const phi = phase(state[1]) - phase(state[0]);

        return new THREE.Vector3(
            Math.sin(theta) * Math.cos(phi),
            Math.sin(theta) * Math.sin(phi),
            Math.cos(theta)
        );
    }, [state]);

    useFrame(() => {
        if (meshRef.current) {
            meshRef.current.rotation.y += 0.002;
        }
    });

    return (
        <group position={position}>
            {/* Sphere */}
            <mesh
                ref={meshRef}
                onPointerOver={() => setHovered(true)}
                onPointerOut={() => setHovered(false)}
            >
                <sphereGeometry args={[1, 32, 32]} />
                <meshPhysicalMaterial
                    color={hovered ? "#8b5cf6" : "#6366f1"}
                    transparent
                    opacity={0.3}
                    roughness={0.1}
                    metalness={0.8}
                    clearcoat={1}
                    clearcoatRoughness={0}
                />
            </mesh>

            {/* Wireframe */}
            <mesh>
                <sphereGeometry args={[1.02, 16, 16]} />
                <meshBasicMaterial color="#4c1d95" wireframe />
            </mesh>

            {/* Axes */}
            <Line points={[[-1.5, 0, 0], [1.5, 0, 0]]} color="#ef4444" lineWidth={1} />
            <Line points={[[0, -1.5, 0], [0, 1.5, 0]]} color="#10b981" lineWidth={1} />
            <Line points={[[0, 0, -1.5], [0, 0, 1.5]]} color="#3b82f6" lineWidth={1} />

            {/* Bloch vector */}
            <Line
                points={[[0, 0, 0], blochVector]}
                color="#fbbf24"
                lineWidth={3}
            />

            {/* Vector tip */}
            <Sphere position={blochVector} args={[0.1]}>
                <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.5} />
            </Sphere>

            {/* Label */}
            <Text
                position={[0, 1.5, 0]}
                fontSize={0.3}
                color="#e5e7eb"
                anchorX="center"
                anchorY="middle"
            >
                {label}
            </Text>

            {/* Quantum state info */}
            {hovered && (
                <Text
                    position={[0, -1.8, 0]}
                    fontSize={0.2}
                    color="#9ca3af"
                    anchorX="center"
                    anchorY="middle"
                >
                    {`|ψ⟩ = ${formatComplex(state[0])}|0⟩ + ${formatComplex(state[1])}|1⟩`}
                </Text>
            )}
        </group>
    );
};

// Entanglement Connection Component
const EntanglementConnection = ({ start, end, strength }: {
    start: THREE.Vector3;
    end: THREE.Vector3;
    strength: number;
}) => {
    const points = useMemo(() => {
        const midPoint = new THREE.Vector3().lerpVectors(start, end, 0.5);
        midPoint.y += 1; // Curve upward

        const curve = new THREE.QuadraticBezierCurve3(start, midPoint, end);
        return curve.getPoints(50);
    }, [start, end]);

    return (
        <Line
            points={points}
            color={d3.interpolateViridis(strength)}
            lineWidth={2 + strength * 3}
            transparent
            opacity={0.6 + strength * 0.4}
        />
    );
};

// Main Bloch Sphere Network Component
export const BlochSphereNetwork = ({ quantumStates, correlations }: {
    quantumStates: QuantumState[];
    correlations: number[][];
}) => {
    const positions = useMemo(() => {
        // Arrange spheres in a circle
        const radius = 5;
        return quantumStates.map((_, i) => {
            const angle = (i / quantumStates.length) * 2 * Math.PI;
            return new THREE.Vector3(
                radius * Math.cos(angle),
                0,
                radius * Math.sin(angle)
            );
        });
    }, [quantumStates.length]);

    return (
        <div className="w-full h-[600px] bg-gray-900 rounded-lg overflow-hidden">
            <Canvas camera={{ position: [0, 8, 15], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <pointLight position={[-10, -10, -10]} intensity={0.5} />

                {/* Bloch spheres */}
                {quantumStates.map((state, i) => (
                    <BlochSphere
                        key={i}
                        position={positions[i]}
                        state={state.amplitude}
                        label={state.label}
                        connections={correlations[i] || []}
                    />
                ))}

                {/* Entanglement connections */}
                {correlations.map((row, i) =>
                    row.map((strength, j) => {
                        if (i < j && strength > 0.3) {
                            return (
                                <EntanglementConnection
                                    key={`${i}-${j}`}
                                    start={positions[i]}
                                    end={positions[j]}
                                    strength={strength}
                                />
                            );
                        }
                        return null;
                    })
                )}

                <OrbitControls enableDamping dampingFactor={0.05} />

                {/* Grid */}
                <gridHelper args={[20, 20, 0x444444, 0x222222]} />
            </Canvas>
        </div>
    );
};

// Wavefunction Collapse Animation Component
export const WavefunctionCollapse = ({ superposition, measurement, onComplete }: {
    superposition: [ComplexNumber, ComplexNumber];
    measurement: number;
    onComplete?: () => void;
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number>();
    const particlesRef = useRef<Array<{
        x: number;
        y: number;
        vx: number;
        vy: number;
        state: number;
        opacity: number;
        size: number;
        color: string;
    }>>([]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;

        // Initialize particles for superposition
        const numParticles = 1000;
        particlesRef.current = Array.from({ length: numParticles }, () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            state: Math.random() < 0.5 ? 0 : 1,
            opacity: 0.5,
            size: Math.random() * 3 + 1,
            color: Math.random() < 0.5 ? '#3b82f6' : '#ef4444'
        }));

        let collapseProgress = 0;
        const collapseSpeed = 0.02;

        const animate = () => {
            ctx.fillStyle = 'rgba(17, 24, 39, 0.1)';
            ctx.fillRect(0, 0, width, height);

            // Update and draw particles
            particlesRef.current.forEach(particle => {
                // Superposition phase - particles move freely
                if (collapseProgress < 0.5) {
                    particle.x += particle.vx;
                    particle.y += particle.vy;

                    // Boundary conditions
                    if (particle.x < 0 || particle.x > width) particle.vx *= -1;
                    if (particle.y < 0 || particle.y > height) particle.vy *= -1;
                } else {
                    // Collapse phase - particles converge to measurement outcome
                    const targetX = measurement === 0 ? width * 0.25 : width * 0.75;
                    const targetY = height * 0.5;

                    const dx = targetX - particle.x;
                    const dy = targetY - particle.y;

                    particle.x += dx * 0.05;
                    particle.y += dy * 0.05;

                    // Fade out particles of wrong state
                    if (particle.state !== measurement) {
                        particle.opacity = Math.max(0, particle.opacity - 0.02);
                    }
                }

                // Draw particle
                ctx.globalAlpha = particle.opacity;
                ctx.fillStyle = particle.color;
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fill();
            });

            // Draw probability clouds
            if (collapseProgress < 0.8) {
                ctx.globalAlpha = 0.2 * (1 - collapseProgress);

                // Left cloud (|0⟩)
                const gradient1 = ctx.createRadialGradient(width * 0.25, height * 0.5, 0, width * 0.25, height * 0.5, 100);
                gradient1.addColorStop(0, '#3b82f6');
                gradient1.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient1;
                ctx.fillRect(0, 0, width * 0.5, height);

                // Right cloud (|1⟩)
                const gradient2 = ctx.createRadialGradient(width * 0.75, height * 0.5, 0, width * 0.75, height * 0.5, 100);
                gradient2.addColorStop(0, '#ef4444');
                gradient2.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient2;
                ctx.fillRect(width * 0.5, 0, width * 0.5, height);
            }

            // Update collapse progress
            if (collapseProgress < 1) {
                collapseProgress += collapseSpeed;
            } else if (onComplete) {
                onComplete();
            }

            animationRef.current = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [measurement, onComplete]);

    return (
        <div className="relative w-full h-[400px] bg-gray-900 rounded-lg overflow-hidden">
            <canvas
                ref={canvasRef}
                width={800}
                height={400}
                className="w-full h-full"
            />
            <div className="absolute top-4 left-4 text-white">
                <h3 className="text-lg font-semibold mb-2">Wavefunction Collapse</h3>
                <div className="text-sm text-gray-300">
                    <div>Initial: |ψ⟩ = {formatComplex(superposition[0])}|0⟩ + {formatComplex(superposition[1])}|1⟩</div>
                    <div>Measurement: |{measurement}⟩</div>
                </div>
            </div>
        </div>
    );
};

// 4D Market Hypercube Visualization
export const MarketHypercube = ({ data, selectedDimensions }: {
    data: MarketDataPoint[];
    selectedDimensions: string[];
}) => {
    const hypercubeRef = useRef<THREE.Group>(null);

    // Project 4D points to 3D
    const project4Dto3D = (point4D: number[]) => {
        const [x, y, z, w] = point4D;
        const distance = 5;
        const wFactor = 1 / (distance - w);

        return new THREE.Vector3(
            x * wFactor,
            y * wFactor,
            z * wFactor
        );
    };

    const HypercubeGeometry = () => {
        const vertices = useMemo(() => {
            // Generate 4D hypercube vertices
            const points: number[][] = [];
            for (let i = 0; i < 16; i++) {
                const x = (i & 1) ? 1 : -1;
                const y = (i & 2) ? 1 : -1;
                const z = (i & 4) ? 1 : -1;
                const w = (i & 8) ? 1 : -1;
                points.push([x, y, z, w]);
            }
            return points;
        }, []);

        const edges = useMemo(() => {
            // Connect vertices that differ by one coordinate
            const connections: [number, number][] = [];
            for (let i = 0; i < 16; i++) {
                for (let j = i + 1; j < 16; j++) {
                    const diff = vertices[i].reduce((acc, val, idx) =>
                        acc + (val !== vertices[j][idx] ? 1 : 0), 0
                    );
                    if (diff === 1) {
                        connections.push([i, j]);
                    }
                }
            }
            return connections;
        }, [vertices]);

        useFrame((state) => {
            if (hypercubeRef.current) {
                // Rotate in 4D space
                const time = state.clock.getElapsedTime();
                hypercubeRef.current.rotation.x = time * 0.1;
                hypercubeRef.current.rotation.y = time * 0.15;
            }
        });

        return (
            <group ref={hypercubeRef}>
                {/* Vertices */}
                {vertices.map((vertex, i) => {
                    const pos3D = project4Dto3D(vertex);
                    const marketData = data[i % data.length];
                    const color = marketData?.value > 0 ? '#10b981' : '#ef4444';

                    return (
                        <mesh key={i} position={pos3D}>
                            <sphereGeometry args={[0.1]} />
                            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
                        </mesh>
                    );
                })}

                {/* Edges */}
                {edges.map(([i, j], idx) => {
                    const start = project4Dto3D(vertices[i]);
                    const end = project4Dto3D(vertices[j]);

                    return (
                        <Line
                            key={idx}
                            points={[start, end]}
                            color="#6366f1"
                            lineWidth={1}
                            transparent
                            opacity={0.6}
                        />
                    );
                })}
            </group>
        );
    };

    return (
        <div className="w-full h-[600px] bg-gray-900 rounded-lg overflow-hidden">
            <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />

                <HypercubeGeometry />

                <OrbitControls enableDamping />

                {/* Dimension labels */}
                <Text position={[3, 0, 0]} fontSize={0.3} color="#ef4444">
                    Price
                </Text>
                <Text position={[0, 3, 0]} fontSize={0.3} color="#10b981">
                    Volatility
                </Text>
                <Text position={[0, 0, 3]} fontSize={0.3} color="#3b82f6">
                    Time
                </Text>

                {/* Grid planes */}
                <gridHelper args={[10, 10, 0x444444, 0x222222]} />
            </Canvas>

            {/* 4th Dimension Slider */}
            <div className="absolute bottom-4 left-4 right-4 bg-gray-800 bg-opacity-90 p-4 rounded-lg">
                <label className="text-white text-sm">4th Dimension: Quantum Confidence</label>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    defaultValue="0.5"
                    className="w-full mt-2"
                    onChange={() => {
                        // Update 4th dimension visualization
                    }}
                />
            </div>
        </div>
    );
};

// Quantum Circuit Live View Component
export const QuantumCircuitLiveView = ({ circuit, gates, isExecuting }: {
    circuit: QuantumCircuit;
    gates: QuantumGate[];
    isExecuting: boolean;
}) => {
    const [highlightedGate, setHighlightedGate] = useState<number | null>(null);
    const [executionProgress, setExecutionProgress] = useState(0);

    useEffect(() => {
        if (isExecuting) {
            const interval = setInterval(() => {
                setExecutionProgress(prev => {
                    if (prev >= gates.length) {
                        clearInterval(interval);
                        return 0;
                    }
                    return prev + 1;
                });
            }, 500);

            return () => clearInterval(interval);
        }
    }, [isExecuting, gates.length]);

    return (
        <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Quantum Circuit Live View</h3>

            <div className="relative">
                {/* Qubit lines */}
                <svg className="w-full h-64">
                    {circuit.qubits.map((qubit, i) => (
                        <g key={i}>
                            <line
                                x1="50"
                                y1={50 + i * 40}
                                x2="750"
                                y2={50 + i * 40}
                                stroke="#4b5563"
                                strokeWidth="2"
                            />
                            <text x="20" y={55 + i * 40} fill="#9ca3af" fontSize="14">
                                q{i}
                            </text>
                        </g>
                    ))}

                    {/* Gates */}
                    {gates.map((gate, i) => {
                        const x = 100 + i * 100;
                        const y = 50 + gate.qubit * 40;
                        const isActive = i === executionProgress - 1;

                        return (
                            <g key={i}>
                                <motion.rect
                                    x={x - 30}
                                    y={y - 15}
                                    width="60"
                                    height="30"
                                    fill={isActive ? "#8b5cf6" : "#374151"}
                                    stroke={isActive ? "#a78bfa" : "#6b7280"}
                                    strokeWidth="2"
                                    rx="4"
                                    initial={{ scale: 0 }}
                                    animate={{ scale: isActive ? 1.1 : 1 }}
                                    transition={{ duration: 0.3 }}
                                    onMouseEnter={() => setHighlightedGate(i)}
                                    onMouseLeave={() => setHighlightedGate(null)}
                                />
                                <text
                                    x={x}
                                    y={y + 5}
                                    textAnchor="middle"
                                    fill="white"
                                    fontSize="12"
                                    fontWeight="bold"
                                >
                                    {gate.type}
                                </text>
                            </g>
                        );
                    })}

                    {/* CNOT connections */}
                    {gates.filter(g => g.type === 'CNOT').map((gate, i) => {
                        const x = 100 + gates.indexOf(gate) * 100;
                        const y1 = 50 + (gate.control || 0) * 40;
                        const y2 = 50 + (gate.target || 0) * 40;

                        return (
                            <g key={`cnot-${i}`}>
                                <line
                                    x1={x}
                                    y1={y1}
                                    x2={x}
                                    y2={y2}
                                    stroke="#8b5cf6"
                                    strokeWidth="2"
                                />
                                <circle cx={x} cy={y1} r="6" fill="#8b5cf6" />
                                <circle cx={x} cy={y2} r="8" fill="none" stroke="#8b5cf6" strokeWidth="2" />
                                <line x1={x - 8} y1={y2} x2={x + 8} y2={y2} stroke="#8b5cf6" strokeWidth="2" />
                                <line x1={x} y1={y2 - 8} x2={x} y2={y2 + 8} stroke="#8b5cf6" strokeWidth="2" />
                            </g>
                        );
                    })}
                </svg>

                {/* Gate details tooltip */}
                {highlightedGate !== null && (
                    <div className="absolute bg-gray-700 p-3 rounded-lg shadow-lg text-white text-sm"
                         style={{
                             left: `${100 + highlightedGate * 100}px`,
                             top: `${50 + gates[highlightedGate].qubit * 40 + 30}px`
                         }}>
                        <div className="font-semibold">{gates[highlightedGate].type} Gate</div>
                        <div className="text-gray-300">Qubit: {gates[highlightedGate].qubit}</div>
                        {gates[highlightedGate].params && (
                            <div className="text-gray-300">
                                Params: {JSON.stringify(gates[highlightedGate].params)}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Execution progress */}
            {isExecuting && (
                <div className="mt-4">
                    <div className="flex justify-between text-sm text-gray-400 mb-1">
                        <span>Executing...</span>
                        <span>{executionProgress}/{gates.length} gates</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                        <motion.div
                            className="bg-purple-500 h-2 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${(executionProgress / gates.length) * 100}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};

export default {
    BlochSphereNetwork,
    WavefunctionCollapse,
    MarketHypercube,
    QuantumCircuitLiveView
};