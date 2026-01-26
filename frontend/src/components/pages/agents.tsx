"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface AgentMessage {
  id: string;
  agent: string;
  type: "thought" | "action" | "observation" | "result";
  content: string;
  timestamp: Date;
}

interface Agent {
  id: string;
  name: string;
  role: "architect" | "analyst" | "judge";
  status: "idle" | "thinking" | "acting" | "waiting";
  color: string;
  description: string;
}

const agents: Agent[] = [
  {
    id: "architect",
    name: "Architect",
    role: "architect",
    status: "idle",
    color: "#6366f1",
    description: "Orchestrates sub-goals and coordinates analyst agents",
  },
  {
    id: "analyst-vol",
    name: "Vol Analyst",
    role: "analyst",
    status: "idle",
    color: "#22c55e",
    description: "Monitors volatility surfaces and options flow",
  },
  {
    id: "analyst-macro",
    name: "Macro Analyst",
    role: "analyst",
    status: "idle",
    color: "#f59e0b",
    description: "Tracks geopolitical events and macro indicators",
  },
  {
    id: "analyst-credit",
    name: "Credit Analyst",
    role: "analyst",
    status: "idle",
    color: "#ec4899",
    description: "Monitors credit spreads and counterparty risk",
  },
  {
    id: "judge",
    name: "Judge",
    role: "judge",
    status: "idle",
    color: "#ef4444",
    description: "Validates actions against compliance rules",
  },
];

export function AgentsPage() {
  const [agentStates, setAgentStates] = useState<Agent[]>(agents);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentTask, setCurrentTask] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Simulate agent activity
  const runAgentDemo = async () => {
    setIsRunning(true);
    setMessages([]);
    setCurrentTask("Analyze current portfolio risk exposure and recommend hedges");

    const addMessage = (
      agent: string,
      type: AgentMessage["type"],
      content: string
    ) => {
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random()}`,
          agent,
          type,
          content,
          timestamp: new Date(),
        },
      ]);
    };

    const setAgentStatus = (agentId: string, status: Agent["status"]) => {
      setAgentStates((prev) =>
        prev.map((a) => (a.id === agentId ? { ...a, status } : a))
      );
    };

    // Architect starts
    setAgentStatus("architect", "thinking");
    await delay(1000);
    addMessage(
      "Architect",
      "thought",
      "Task received: Analyze portfolio risk. I need to gather information from multiple sources."
    );
    await delay(1500);

    addMessage(
      "Architect",
      "thought",
      "Breaking down into sub-goals: 1) Check vol surface, 2) Review macro conditions, 3) Assess credit risk"
    );
    await delay(1000);

    setAgentStatus("architect", "acting");
    addMessage(
      "Architect",
      "action",
      "Dispatching Vol Analyst to check current implied volatility levels"
    );

    // Vol Analyst
    setAgentStatus("architect", "waiting");
    setAgentStatus("analyst-vol", "thinking");
    await delay(1200);
    addMessage(
      "Vol Analyst",
      "thought",
      "Querying KDB-X for current vol surface and historical percentiles..."
    );

    setAgentStatus("analyst-vol", "acting");
    await delay(1500);
    addMessage(
      "Vol Analyst",
      "observation",
      "SPY 30d IV at 18.5% (45th percentile). VIX term structure in contango. 25Δ skew elevated at +3.2%."
    );
    await delay(800);
    addMessage(
      "Vol Analyst",
      "result",
      "Vol environment: NEUTRAL with slight fear skew. Put protection relatively expensive."
    );
    setAgentStatus("analyst-vol", "idle");

    // Macro Analyst
    setAgentStatus("architect", "acting");
    await delay(500);
    addMessage(
      "Architect",
      "action",
      "Dispatching Macro Analyst to assess geopolitical conditions"
    );

    setAgentStatus("architect", "waiting");
    setAgentStatus("analyst-macro", "thinking");
    await delay(1000);
    addMessage(
      "Macro Analyst",
      "thought",
      "Scanning FinGPT semantic signals from Fed communications and news feeds..."
    );

    setAgentStatus("analyst-macro", "acting");
    await delay(1800);
    addMessage(
      "Macro Analyst",
      "observation",
      "Fed minutes sentiment: -0.15 (slightly hawkish). Treasury yield curve steepening. No major geopolitical events detected."
    );
    await delay(600);
    addMessage(
      "Macro Analyst",
      "result",
      "Macro environment: CAUTIOUS. Rate sensitivity elevated. Recommend duration hedges."
    );
    setAgentStatus("analyst-macro", "idle");

    // Credit Analyst
    setAgentStatus("architect", "acting");
    await delay(500);
    addMessage(
      "Architect",
      "action",
      "Dispatching Credit Analyst to run contagion simulation"
    );

    setAgentStatus("architect", "waiting");
    setAgentStatus("analyst-credit", "thinking");
    await delay(1200);
    addMessage(
      "Credit Analyst",
      "thought",
      "Running GNN contagion model on counterparty network..."
    );

    setAgentStatus("analyst-credit", "acting");
    await delay(2000);
    addMessage(
      "Credit Analyst",
      "observation",
      "Prime broker exposure within limits. CDS spreads stable. Contagion risk score: 0.23 (LOW)."
    );
    await delay(500);
    addMessage(
      "Credit Analyst",
      "result",
      "Credit environment: STABLE. No immediate counterparty concerns."
    );
    setAgentStatus("analyst-credit", "idle");

    // Architect synthesizes
    setAgentStatus("architect", "thinking");
    await delay(1500);
    addMessage(
      "Architect",
      "thought",
      "Synthesizing analyst reports. Preparing hedge recommendations..."
    );

    setAgentStatus("architect", "acting");
    await delay(1000);
    addMessage(
      "Architect",
      "action",
      "Running THRML thermodynamic optimizer for optimal hedge basket"
    );

    await delay(2000);
    addMessage(
      "Architect",
      "result",
      "RECOMMENDATION: 1) Buy SPY 390 puts (2% portfolio), 2) Add TLT position (5% portfolio), 3) Maintain current equity exposure"
    );

    // Judge validates
    setAgentStatus("architect", "waiting");
    setAgentStatus("judge", "thinking");
    await delay(800);
    addMessage(
      "Judge",
      "thought",
      "Validating proposed trades against compliance ruleset..."
    );

    setAgentStatus("judge", "acting");
    await delay(1200);
    addMessage(
      "Judge",
      "observation",
      "Checking: Position limits ✓, Restricted list ✓, Leverage constraints ✓, Liquidity requirements ✓"
    );
    await delay(500);
    addMessage(
      "Judge",
      "result",
      "APPROVED: All proposed trades pass compliance checks. Ready for execution."
    );
    setAgentStatus("judge", "idle");

    // Complete
    setAgentStatus("architect", "idle");
    await delay(500);
    addMessage(
      "Architect",
      "result",
      "Task complete. Hedge recommendations validated and ready for human review."
    );

    setIsRunning(false);
  };

  const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const messageColors: Record<AgentMessage["type"], string> = {
    thought: "text-blue-400",
    action: "text-amber-400",
    observation: "text-green-400",
    result: "text-purple-400",
  };

  const messageIcons: Record<AgentMessage["type"], string> = {
    thought: "💭",
    action: "⚡",
    observation: "👁️",
    result: "✅",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Agentic AI Risk Swarm
          </h1>
          <p className="text-muted-foreground">
            Autonomous risk monitoring with ReAct pattern and Dual LLM security
          </p>
        </div>

        <button
          onClick={runAgentDemo}
          disabled={isRunning}
          className={cn(
            "px-6 py-2.5 rounded-lg font-medium transition-all",
            isRunning
              ? "bg-muted text-muted-foreground cursor-not-allowed"
              : "bg-gradient-to-r from-indigo-500 to-purple-500 text-white hover:opacity-90 shadow-lg"
          )}
        >
          {isRunning ? "Running..." : "Run Agent Demo"}
        </button>
      </div>

      {/* Agent status cards */}
      <div className="grid grid-cols-5 gap-4">
        {agentStates.map((agent) => (
          <Card
            key={agent.id}
            className={cn(
              "bg-card/50 border-border/50 transition-all duration-300",
              agent.status !== "idle" && "ring-2",
              agent.status === "thinking" && "ring-blue-500/50",
              agent.status === "acting" && "ring-amber-500/50",
              agent.status === "waiting" && "ring-gray-500/50"
            )}
          >
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <div
                  className={cn(
                    "w-2 h-2 rounded-full",
                    agent.status === "idle" && "bg-gray-500",
                    agent.status === "thinking" && "bg-blue-500 animate-pulse",
                    agent.status === "acting" && "bg-amber-500 animate-pulse",
                    agent.status === "waiting" && "bg-gray-400"
                  )}
                />
                <span style={{ color: agent.color }}>{agent.name}</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-muted-foreground">
                {agent.description}
              </div>
              <div className="mt-2 text-xs font-medium capitalize">
                Status:{" "}
                <span
                  className={cn(
                    agent.status === "thinking" && "text-blue-400",
                    agent.status === "acting" && "text-amber-400",
                    agent.status === "waiting" && "text-gray-400",
                    agent.status === "idle" && "text-green-400"
                  )}
                >
                  {agent.status}
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Current task */}
      {currentTask && (
        <Card className="bg-card/30 border-border/50">
          <CardContent className="py-4">
            <div className="flex items-center gap-3">
              <div className="text-sm font-medium text-muted-foreground">
                Current Task:
              </div>
              <div className="text-sm text-white">{currentTask}</div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Agent message log */}
      <Card className="bg-card/30 border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Agent Activity Log</span>
            <div className="flex items-center gap-4 text-xs font-normal">
              {Object.entries(messageIcons).map(([type, icon]) => (
                <span key={type} className="flex items-center gap-1">
                  <span>{icon}</span>
                  <span className={messageColors[type as AgentMessage["type"]]}>
                    {type}
                  </span>
                </span>
              ))}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] overflow-y-auto space-y-3 font-mono text-sm">
            {messages.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                Click "Run Agent Demo" to see the Risk Swarm in action
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className="flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300"
                >
                  <span className="text-lg">
                    {messageIcons[msg.type]}
                  </span>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span
                        className="font-semibold"
                        style={{
                          color: agentStates.find((a) =>
                            a.name === msg.agent
                          )?.color,
                        }}
                      >
                        {msg.agent}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {msg.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <div className={cn("mt-0.5", messageColors[msg.type])}>
                      {msg.content}
                    </div>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </CardContent>
      </Card>

      {/* Architecture diagram */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>ReAct Pattern</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 text-sm">
              <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <div className="font-semibold text-blue-400 mb-1">
                  1. Thought
                </div>
                <div className="text-muted-foreground">
                  Agent reasons about the current state and decides next action
                </div>
              </div>
              <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                <div className="font-semibold text-amber-400 mb-1">
                  2. Action
                </div>
                <div className="text-muted-foreground">
                  Agent executes tool call (query KDB-X, run THRML, etc.)
                </div>
              </div>
              <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                <div className="font-semibold text-green-400 mb-1">
                  3. Observation
                </div>
                <div className="text-muted-foreground">
                  Agent receives and interprets tool output
                </div>
              </div>
              <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <div className="font-semibold text-purple-400 mb-1">
                  4. Result / Loop
                </div>
                <div className="text-muted-foreground">
                  Agent produces result or continues reasoning
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Dual LLM Security Pattern</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 text-sm">
              <div className="p-4 rounded-lg bg-muted/30">
                <div className="font-semibold mb-2 text-amber-400">
                  Untrusted Layer (Reader LLM)
                </div>
                <ul className="list-disc list-inside text-muted-foreground space-y-1">
                  <li>Processes raw external text (news, social)</li>
                  <li>No access to trading tools</li>
                  <li>Outputs sanitized JSON summary</li>
                </ul>
              </div>
              <div className="flex items-center justify-center">
                <div className="px-4 py-2 rounded bg-red-500/20 text-red-400 text-xs font-mono">
                  🛡️ Security Boundary
                </div>
              </div>
              <div className="p-4 rounded-lg bg-muted/30">
                <div className="font-semibold mb-2 text-green-400">
                  Trusted Layer (Thinker LLM)
                </div>
                <ul className="list-disc list-inside text-muted-foreground space-y-1">
                  <li>Receives only sanitized JSON</li>
                  <li>Has access to trading tools</li>
                  <li>Cannot see raw malicious text</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
