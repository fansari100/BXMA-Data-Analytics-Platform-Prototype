"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Users,
  Building2,
  Scale,
  DollarSign,
  ShieldCheck,
  Bell,
  CheckCircle,
  Clock,
  AlertTriangle,
  Send,
  MessageSquare,
  FileText,
  PlusCircle,
} from "lucide-react";

// Team types matching backend
type TeamType = "INVESTMENT" | "OPERATIONS" | "TREASURY" | "LEGAL" | "RISK_QUANT";

interface TeamMember {
  user_id: string;
  name: string;
  email: string;
  team: TeamType;
  title: string;
  is_manager: boolean;
}

interface WorkflowTask {
  task_id: string;
  title: string;
  description: string;
  task_type: string;
  created_by: string;
  assigned_to: string;
  team: string;
  status: "DRAFT" | "PENDING" | "IN_PROGRESS" | "REVIEW" | "APPROVED" | "REJECTED" | "COMPLETED";
  priority: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  created_at: string;
  due_date: string | null;
}

interface Notification {
  notification_id: string;
  title: string;
  message: string;
  notification_type: string;
  recipient_id: string;
  priority: string;
  created_at: string;
  read_at: string | null;
}

const teamConfig: Record<TeamType, { icon: React.ElementType; color: string; label: string }> = {
  INVESTMENT: { icon: DollarSign, color: "text-accent-emerald", label: "Investment Teams" },
  OPERATIONS: { icon: Building2, color: "text-accent-cyan", label: "Operations" },
  TREASURY: { icon: DollarSign, color: "text-accent-amber", label: "Treasury" },
  LEGAL: { icon: Scale, color: "text-accent-violet", label: "Legal & Compliance" },
  RISK_QUANT: { icon: ShieldCheck, color: "text-accent-rose", label: "Data Analytics" },
};

const statusColors: Record<string, string> = {
  DRAFT: "bg-dark-600 text-dark-200",
  PENDING: "bg-amber-500/20 text-amber-400",
  IN_PROGRESS: "bg-cyan-500/20 text-cyan-400",
  REVIEW: "bg-violet-500/20 text-violet-400",
  APPROVED: "bg-emerald-500/20 text-emerald-400",
  REJECTED: "bg-rose-500/20 text-rose-400",
  COMPLETED: "bg-emerald-500/20 text-emerald-400",
};

const priorityColors: Record<string, string> = {
  CRITICAL: "bg-rose-500/20 text-rose-400 border-rose-500/50",
  HIGH: "bg-amber-500/20 text-amber-400 border-amber-500/50",
  MEDIUM: "bg-cyan-500/20 text-cyan-400 border-cyan-500/50",
  LOW: "bg-dark-600/50 text-dark-300 border-dark-500",
};

// Sample data (in production, fetch from API)
const sampleTeamMembers: TeamMember[] = [
  { user_id: "INV001", name: "John Smith", email: "john.smith@blackstone.com", team: "INVESTMENT", title: "Senior Portfolio Manager", is_manager: true },
  { user_id: "OPS001", name: "Jane Doe", email: "jane.doe@blackstone.com", team: "OPERATIONS", title: "Operations Manager", is_manager: true },
  { user_id: "TRE001", name: "Bob Johnson", email: "bob.johnson@blackstone.com", team: "TREASURY", title: "Treasury Director", is_manager: true },
  { user_id: "LEG001", name: "Alice Williams", email: "alice.williams@blackstone.com", team: "LEGAL", title: "Chief Compliance Officer", is_manager: true },
  { user_id: "RQ001", name: "Farooq Ansari", email: "farooq.ansari@blackstone.com", team: "RISK_QUANT", title: "Risk Analyst", is_manager: false },
];

const sampleTasks: WorkflowTask[] = [
  { task_id: "T001", title: "Update VaR Limit for Equity Portfolio", description: "Review and approve new VaR limits", task_type: "risk_limit_change", created_by: "RQ001", assigned_to: "INV001", team: "INVESTMENT", status: "REVIEW", priority: "HIGH", created_at: "2026-01-25T10:00:00Z", due_date: "2026-01-27" },
  { task_id: "T002", title: "Factor Model Validation Q1 2026", description: "Validate updated factor model", task_type: "model_update", created_by: "RQ001", assigned_to: "LEG001", team: "LEGAL", status: "PENDING", priority: "MEDIUM", created_at: "2026-01-24T15:00:00Z", due_date: "2026-01-31" },
  { task_id: "T003", title: "Treasury Cash Flow Report", description: "Weekly cash flow projection", task_type: "report_request", created_by: "TRE001", assigned_to: "RQ001", team: "RISK_QUANT", status: "IN_PROGRESS", priority: "HIGH", created_at: "2026-01-25T09:00:00Z", due_date: "2026-01-26" },
];

const sampleNotifications: Notification[] = [
  { notification_id: "N001", title: "VaR Limit Breach Alert", message: "Portfolio XYZ has exceeded 95% VaR limit", notification_type: "LIMIT_BREACH", recipient_id: "RQ001", priority: "CRITICAL", created_at: "2026-01-25T14:30:00Z", read_at: null },
  { notification_id: "N002", title: "Approval Required", message: "Risk limit change request awaiting your approval", notification_type: "APPROVAL_REQUIRED", recipient_id: "RQ001", priority: "HIGH", created_at: "2026-01-25T12:00:00Z", read_at: null },
  { notification_id: "N003", title: "Report Ready", message: "Daily risk report is ready for download", notification_type: "REPORT_READY", recipient_id: "RQ001", priority: "LOW", created_at: "2026-01-25T08:00:00Z", read_at: "2026-01-25T09:15:00Z" },
];

export function CollaborationPage() {
  const [selectedTeam, setSelectedTeam] = useState<TeamType | "ALL">("ALL");
  const [tasks, setTasks] = useState<WorkflowTask[]>(sampleTasks);
  const [notifications, setNotifications] = useState<Notification[]>(sampleNotifications);
  const [showNewTaskForm, setShowNewTaskForm] = useState(false);
  const [newTask, setNewTask] = useState({
    title: "",
    description: "",
    task_type: "data_request",
    assigned_to: "",
    priority: "MEDIUM",
  });

  const unreadCount = notifications.filter((n) => !n.read_at).length;
  
  const filteredTasks = selectedTeam === "ALL" 
    ? tasks 
    : tasks.filter((t) => t.team === selectedTeam || sampleTeamMembers.find((m) => m.user_id === t.assigned_to)?.team === selectedTeam);

  const handleCreateTask = () => {
    const task: WorkflowTask = {
      task_id: `T${String(tasks.length + 1).padStart(3, "0")}`,
      ...newTask,
      created_by: "RQ001",
      team: sampleTeamMembers.find((m) => m.user_id === newTask.assigned_to)?.team || "RISK_QUANT",
      status: "PENDING",
      priority: newTask.priority as any,
      created_at: new Date().toISOString(),
      due_date: null,
    };
    setTasks([task, ...tasks]);
    setShowNewTaskForm(false);
    setNewTask({ title: "", description: "", task_type: "data_request", assigned_to: "", priority: "MEDIUM" });
  };

  const markNotificationRead = (id: string) => {
    setNotifications(notifications.map((n) =>
      n.notification_id === id ? { ...n, read_at: new Date().toISOString() } : n
    ));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-dark-100">Cross-Team Collaboration</h1>
          <p className="text-dark-400 mt-1">
            Seamless integration across Investment, Operations, Treasury, and Legal teams
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="relative">
            <Bell className="w-6 h-6 text-dark-300 cursor-pointer hover:text-dark-100 transition-colors" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-accent-rose text-dark-950 text-xs rounded-full flex items-center justify-center">
                {unreadCount}
              </span>
            )}
          </div>
          <Button
            onClick={() => setShowNewTaskForm(true)}
            className="bg-accent-cyan hover:bg-accent-cyan/80 text-dark-950"
          >
            <PlusCircle className="w-4 h-4 mr-2" />
            New Task
          </Button>
        </div>
      </div>

      {/* Team Filter */}
      <div className="flex gap-2 flex-wrap">
        <Button
          variant={selectedTeam === "ALL" ? "default" : "outline"}
          onClick={() => setSelectedTeam("ALL")}
          className={selectedTeam === "ALL" ? "bg-accent-cyan text-dark-950" : "border-dark-600 text-dark-300"}
        >
          <Users className="w-4 h-4 mr-2" />
          All Teams
        </Button>
        {(Object.keys(teamConfig) as TeamType[]).map((team) => {
          const config = teamConfig[team];
          const Icon = config.icon;
          return (
            <Button
              key={team}
              variant={selectedTeam === team ? "default" : "outline"}
              onClick={() => setSelectedTeam(team)}
              className={selectedTeam === team ? "bg-accent-cyan text-dark-950" : "border-dark-600 text-dark-300"}
            >
              <Icon className={`w-4 h-4 mr-2 ${config.color}`} />
              {config.label}
            </Button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Notifications Panel */}
        <Card className="bg-dark-800/50 border-dark-700">
          <CardHeader className="border-b border-dark-700">
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <Bell className="w-5 h-5 text-accent-amber" />
              Notifications
              {unreadCount > 0 && (
                <span className="bg-accent-rose text-dark-950 text-xs px-2 py-0.5 rounded-full">
                  {unreadCount} new
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 max-h-[400px] overflow-y-auto">
            <AnimatePresence>
              {notifications.map((notification, idx) => (
                <motion.div
                  key={notification.notification_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`p-4 border-b border-dark-700 cursor-pointer hover:bg-dark-700/50 transition-colors ${
                    !notification.read_at ? "bg-dark-750" : ""
                  }`}
                  onClick={() => markNotificationRead(notification.notification_id)}
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-2 h-2 rounded-full mt-2 ${
                      notification.priority === "CRITICAL" ? "bg-rose-500" :
                      notification.priority === "HIGH" ? "bg-amber-500" : "bg-dark-500"
                    }`} />
                    <div className="flex-1 min-w-0">
                      <p className="text-dark-100 font-medium text-sm truncate">
                        {notification.title}
                      </p>
                      <p className="text-dark-400 text-xs mt-1 line-clamp-2">
                        {notification.message}
                      </p>
                      <p className="text-dark-500 text-xs mt-2">
                        {new Date(notification.created_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </CardContent>
        </Card>

        {/* Tasks Panel */}
        <Card className="bg-dark-800/50 border-dark-700 lg:col-span-2">
          <CardHeader className="border-b border-dark-700">
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <FileText className="w-5 h-5 text-accent-cyan" />
              Active Tasks
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 max-h-[400px] overflow-y-auto">
            <AnimatePresence>
              {filteredTasks.map((task, idx) => {
                const assignee = sampleTeamMembers.find((m) => m.user_id === task.assigned_to);
                return (
                  <motion.div
                    key={task.task_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="p-4 border-b border-dark-700 hover:bg-dark-700/30 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColors[task.status]}`}>
                            {task.status.replace("_", " ")}
                          </span>
                          <span className={`px-2 py-0.5 rounded text-xs border ${priorityColors[task.priority]}`}>
                            {task.priority}
                          </span>
                        </div>
                        <p className="text-dark-100 font-medium mt-2">{task.title}</p>
                        <p className="text-dark-400 text-sm mt-1">{task.description}</p>
                        <div className="flex items-center gap-4 mt-3 text-xs text-dark-500">
                          {assignee && (
                            <span className="flex items-center gap-1">
                              <Users className="w-3 h-3" />
                              {assignee.name} ({assignee.team})
                            </span>
                          )}
                          {task.due_date && (
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              Due: {task.due_date}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {task.status === "REVIEW" && (
                          <>
                            <Button size="sm" className="bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30">
                              <CheckCircle className="w-4 h-4" />
                            </Button>
                            <Button size="sm" className="bg-rose-500/20 text-rose-400 hover:bg-rose-500/30">
                              <AlertTriangle className="w-4 h-4" />
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </CardContent>
        </Card>
      </div>

      {/* Team Directory */}
      <Card className="bg-dark-800/50 border-dark-700">
        <CardHeader>
          <CardTitle className="text-dark-100">Team Directory</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {(Object.keys(teamConfig) as TeamType[]).map((team) => {
              const config = teamConfig[team];
              const Icon = config.icon;
              const members = sampleTeamMembers.filter((m) => m.team === team);
              
              return (
                <motion.div
                  key={team}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-dark-700/50 rounded-lg p-4"
                >
                  <div className="flex items-center gap-2 mb-3">
                    <Icon className={`w-5 h-5 ${config.color}`} />
                    <span className="text-dark-100 font-medium text-sm">{config.label}</span>
                  </div>
                  <div className="space-y-2">
                    {members.map((member) => (
                      <div key={member.user_id} className="text-xs">
                        <p className="text-dark-200">{member.name}</p>
                        <p className="text-dark-500">{member.title}</p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* New Task Modal */}
      <AnimatePresence>
        {showNewTaskForm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
            onClick={() => setShowNewTaskForm(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-dark-800 border border-dark-700 rounded-xl p-6 w-full max-w-md"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-xl font-bold text-dark-100 mb-4">Create New Task</h3>
              <div className="space-y-4">
                <div>
                  <Label className="text-dark-300">Title</Label>
                  <Input
                    value={newTask.title}
                    onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                    className="bg-dark-900 border-dark-600 text-dark-100 mt-1"
                    placeholder="Task title"
                  />
                </div>
                <div>
                  <Label className="text-dark-300">Description</Label>
                  <Textarea
                    value={newTask.description}
                    onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                    className="bg-dark-900 border-dark-600 text-dark-100 mt-1"
                    placeholder="Task description"
                    rows={3}
                  />
                </div>
                <div>
                  <Label className="text-dark-300">Assign To</Label>
                  <select
                    value={newTask.assigned_to}
                    onChange={(e) => setNewTask({ ...newTask, assigned_to: e.target.value })}
                    className="w-full bg-dark-900 border border-dark-600 text-dark-100 rounded-md p-2 mt-1"
                  >
                    <option value="">Select team member</option>
                    {sampleTeamMembers.map((member) => (
                      <option key={member.user_id} value={member.user_id}>
                        {member.name} ({member.team})
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <Label className="text-dark-300">Priority</Label>
                  <select
                    value={newTask.priority}
                    onChange={(e) => setNewTask({ ...newTask, priority: e.target.value })}
                    className="w-full bg-dark-900 border border-dark-600 text-dark-100 rounded-md p-2 mt-1"
                  >
                    <option value="LOW">Low</option>
                    <option value="MEDIUM">Medium</option>
                    <option value="HIGH">High</option>
                    <option value="CRITICAL">Critical</option>
                  </select>
                </div>
                <div className="flex gap-2 mt-6">
                  <Button
                    variant="outline"
                    className="flex-1 border-dark-600 text-dark-300"
                    onClick={() => setShowNewTaskForm(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    className="flex-1 bg-accent-cyan text-dark-950"
                    onClick={handleCreateTask}
                    disabled={!newTask.title || !newTask.assigned_to}
                  >
                    <Send className="w-4 h-4 mr-2" />
                    Create Task
                  </Button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
