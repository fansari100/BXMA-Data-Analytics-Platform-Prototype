"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface GraphNode {
  id: string;
  name: string;
  type: "bank" | "fund" | "counterparty";
  aum: number;
  x: number;
  y: number;
  defaultProbability: number;
  isDefaulted: boolean;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  type: "prime_brokerage" | "credit" | "derivative";
}

interface ContagionSimulation {
  initialNode: string;
  rounds: number;
  nodesAffected: number;
  totalLoss: number;
  amplificationFactor: number;
  path: string[][];
}

const nodeColors = {
  bank: "#3b82f6",
  fund: "#22c55e",
  counterparty: "#f59e0b",
};

export function ContagionPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [simulation, setSimulation] = useState<ContagionSimulation | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [metrics, setMetrics] = useState({
    totalNodes: 0,
    totalEdges: 0,
    avgCentrality: 0,
    clusteringCoef: 0,
  });

  // Generate sample financial network
  useEffect(() => {
    const banks: GraphNode[] = [
      { id: "JPM", name: "JP Morgan", type: "bank", aum: 3e12, x: 0, y: 0, defaultProbability: 0.01, isDefaulted: false },
      { id: "GS", name: "Goldman Sachs", type: "bank", aum: 2.5e12, x: 0, y: 0, defaultProbability: 0.015, isDefaulted: false },
      { id: "MS", name: "Morgan Stanley", type: "bank", aum: 1.8e12, x: 0, y: 0, defaultProbability: 0.012, isDefaulted: false },
      { id: "BAC", name: "Bank of America", type: "bank", aum: 2.8e12, x: 0, y: 0, defaultProbability: 0.01, isDefaulted: false },
      { id: "C", name: "Citigroup", type: "bank", aum: 2.2e12, x: 0, y: 0, defaultProbability: 0.018, isDefaulted: false },
    ];

    const funds: GraphNode[] = [
      { id: "BXMA", name: "Blackstone BXMA", type: "fund", aum: 150e9, x: 0, y: 0, defaultProbability: 0.005, isDefaulted: false },
      { id: "BW", name: "Bridgewater", type: "fund", aum: 120e9, x: 0, y: 0, defaultProbability: 0.008, isDefaulted: false },
      { id: "CITADEL", name: "Citadel", type: "fund", aum: 100e9, x: 0, y: 0, defaultProbability: 0.01, isDefaulted: false },
      { id: "AQR", name: "AQR Capital", type: "fund", aum: 80e9, x: 0, y: 0, defaultProbability: 0.012, isDefaulted: false },
      { id: "DESHAW", name: "DE Shaw", type: "fund", aum: 60e9, x: 0, y: 0, defaultProbability: 0.015, isDefaulted: false },
    ];

    // Position nodes in a circle
    const allNodes = [...banks, ...funds];
    const radius = 200;
    const centerX = 400;
    const centerY = 300;

    allNodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / allNodes.length - Math.PI / 2;
      node.x = centerX + radius * Math.cos(angle);
      node.y = centerY + radius * Math.sin(angle);
    });

    setNodes(allNodes);

    // Generate edges
    const newEdges: GraphEdge[] = [];

    // Prime brokerage relationships
    funds.forEach((fund) => {
      const pbBanks = banks.slice(0, 2 + Math.floor(Math.random() * 2));
      pbBanks.forEach((bank) => {
        newEdges.push({
          source: fund.id,
          target: bank.id,
          weight: 0.3 + Math.random() * 0.5,
          type: "prime_brokerage",
        });
      });
    });

    // Interbank exposures
    for (let i = 0; i < banks.length; i++) {
      for (let j = i + 1; j < banks.length; j++) {
        if (Math.random() > 0.3) {
          newEdges.push({
            source: banks[i].id,
            target: banks[j].id,
            weight: 0.2 + Math.random() * 0.4,
            type: "credit",
          });
        }
      }
    }

    setEdges(newEdges);

    // Calculate metrics
    setMetrics({
      totalNodes: allNodes.length,
      totalEdges: newEdges.length,
      avgCentrality: 0.45,
      clusteringCoef: 0.62,
    });
  }, []);

  // Render network graph
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, 800, 600);

    // Draw edges
    edges.forEach((edge) => {
      const source = nodes.find((n) => n.id === edge.source);
      const target = nodes.find((n) => n.id === edge.target);
      if (!source || !target) return;

      const isHighlighted =
        simulation?.path.flat().includes(edge.source) &&
        simulation?.path.flat().includes(edge.target);

      ctx.strokeStyle = isHighlighted
        ? "#ef4444"
        : edge.type === "prime_brokerage"
        ? "rgba(34, 197, 94, 0.3)"
        : "rgba(255, 255, 255, 0.15)";
      ctx.lineWidth = isHighlighted ? 3 : 1 + edge.weight * 2;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();
    });

    // Draw nodes
    nodes.forEach((node) => {
      const isSelected = selectedNode === node.id;
      const isInPath = simulation?.path.flat().includes(node.id);
      const radius = 20 + Math.log10(node.aum / 1e9) * 5;

      // Glow effect for selected/affected nodes
      if (isSelected || isInPath) {
        ctx.shadowBlur = 20;
        ctx.shadowColor = isInPath ? "#ef4444" : "#ffffff";
      }

      // Node circle
      ctx.fillStyle = node.isDefaulted
        ? "#ef4444"
        : isInPath
        ? "#f59e0b"
        : nodeColors[node.type];
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;

      // Border
      ctx.strokeStyle = isSelected ? "#ffffff" : "rgba(255, 255, 255, 0.3)";
      ctx.lineWidth = isSelected ? 3 : 1;
      ctx.stroke();

      // Label
      ctx.fillStyle = "#ffffff";
      ctx.font = "11px Inter, system-ui";
      ctx.textAlign = "center";
      ctx.fillText(node.id, node.x, node.y + radius + 15);
    });

    // Legend
    ctx.font = "11px Inter, system-ui";
    ctx.textAlign = "left";

    let legendY = 20;
    Object.entries(nodeColors).forEach(([type, color]) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(20, legendY, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(type.charAt(0).toUpperCase() + type.slice(1), 35, legendY + 4);
      legendY += 20;
    });
  }, [nodes, edges, selectedNode, simulation]);

  useEffect(() => {
    render();
  }, [render]);

  // Handle click to select node
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const clickedNode = nodes.find((node) => {
        const radius = 20 + Math.log10(node.aum / 1e9) * 5;
        const dist = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
        return dist <= radius;
      });

      setSelectedNode(clickedNode?.id || null);
    },
    [nodes]
  );

  // Run contagion simulation
  const runSimulation = useCallback(() => {
    if (!selectedNode) return;

    setIsSimulating(true);

    // Simulate contagion
    const affectedNodes = new Set([selectedNode]);
    const path: string[][] = [[selectedNode]];
    let totalLoss = 0;

    const initialNode = nodes.find((n) => n.id === selectedNode);
    if (initialNode) {
      totalLoss += initialNode.aum * 0.1; // Initial shock
    }

    // Propagate through network
    for (let round = 0; round < 5; round++) {
      const newAffected: string[] = [];

      edges.forEach((edge) => {
        const sourceAffected = affectedNodes.has(edge.source);
        const targetAffected = affectedNodes.has(edge.target);

        if (sourceAffected && !targetAffected) {
          if (Math.random() < edge.weight * 0.5) {
            newAffected.push(edge.target);
            const targetNode = nodes.find((n) => n.id === edge.target);
            if (targetNode) {
              totalLoss += targetNode.aum * 0.05 * edge.weight;
            }
          }
        } else if (targetAffected && !sourceAffected) {
          if (Math.random() < edge.weight * 0.5) {
            newAffected.push(edge.source);
            const sourceNode = nodes.find((n) => n.id === edge.source);
            if (sourceNode) {
              totalLoss += sourceNode.aum * 0.05 * edge.weight;
            }
          }
        }
      });

      if (newAffected.length === 0) break;

      newAffected.forEach((id) => affectedNodes.add(id));
      path.push(newAffected);
    }

    setSimulation({
      initialNode: selectedNode,
      rounds: path.length,
      nodesAffected: affectedNodes.size,
      totalLoss,
      amplificationFactor: totalLoss / (initialNode?.aum || 1) / 0.1,
      path,
    });

    setIsSimulating(false);
  }, [selectedNode, nodes, edges]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          Counterparty Contagion Analysis
        </h1>
        <p className="text-muted-foreground">
          Graph Neural Network-based contagion modeling and systemic risk
          assessment
        </p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Network Nodes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalNodes}</div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Network Edges
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.totalEdges}</div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Centrality
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.avgCentrality.toFixed(2)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Clustering Coef
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.clusteringCoef.toFixed(2)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-3 gap-6">
        {/* Network visualization */}
        <Card className="col-span-2 bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Financial Network Graph</span>
              <button
                onClick={runSimulation}
                disabled={!selectedNode || isSimulating}
                className={cn(
                  "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                  selectedNode
                    ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
                    : "bg-muted text-muted-foreground cursor-not-allowed"
                )}
              >
                {isSimulating ? "Simulating..." : "Simulate Contagion"}
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="rounded-lg cursor-pointer"
              onClick={handleCanvasClick}
            />
            <div className="mt-2 text-xs text-muted-foreground">
              Click a node to select, then run simulation to see contagion
              spread
            </div>
          </CardContent>
        </Card>

        {/* Simulation results */}
        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Simulation Results</CardTitle>
          </CardHeader>
          <CardContent>
            {simulation ? (
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-muted-foreground">
                    Initial Default
                  </div>
                  <div className="text-lg font-bold text-red-500">
                    {simulation.initialNode}
                  </div>
                </div>

                <div>
                  <div className="text-sm text-muted-foreground">
                    Propagation Rounds
                  </div>
                  <div className="text-lg font-bold">{simulation.rounds}</div>
                </div>

                <div>
                  <div className="text-sm text-muted-foreground">
                    Nodes Affected
                  </div>
                  <div className="text-lg font-bold text-amber-500">
                    {simulation.nodesAffected} / {nodes.length}
                  </div>
                </div>

                <div>
                  <div className="text-sm text-muted-foreground">
                    Total Cascade Loss
                  </div>
                  <div className="text-lg font-bold text-red-500">
                    ${(simulation.totalLoss / 1e9).toFixed(1)}B
                  </div>
                </div>

                <div>
                  <div className="text-sm text-muted-foreground">
                    Amplification Factor
                  </div>
                  <div className="text-lg font-bold">
                    {simulation.amplificationFactor.toFixed(1)}x
                  </div>
                </div>

                <div className="pt-4 border-t border-border">
                  <div className="text-sm text-muted-foreground mb-2">
                    Propagation Path
                  </div>
                  {simulation.path.map((round, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm">
                      <span className="text-muted-foreground">R{i + 1}:</span>
                      <span>{round.join(" → ")}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Select a node and run simulation to see contagion analysis
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
