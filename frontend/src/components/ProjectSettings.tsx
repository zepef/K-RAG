import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { FolderOpen, Save } from "lucide-react";

interface ProjectInfo {
  id: string;
  name: string;
  created_at?: string;
  session_count: number;
  last_updated?: string;
}

interface ProjectSettingsProps {
  projectName: string;
  projectId: string;
  savePrompts: boolean;
  onProjectNameChange: (name: string) => void;
  onSavePromptsToggle: (enabled: boolean) => void;
  onNewProject: () => void;
  onSelectProject: (projectId: string, projectName: string) => void;
}

export const ProjectSettings: React.FC<ProjectSettingsProps> = ({
  projectName,
  projectId,
  savePrompts,
  onProjectNameChange,
  onSavePromptsToggle,
  onNewProject,
  onSelectProject,
}) => {
  const [existingProjects, setExistingProjects] = useState<ProjectInfo[]>([]);
  const [_isLoadingProjects, setIsLoadingProjects] = useState(false);
  const [selectedMode, setSelectedMode] = useState<"new" | "existing">("new");
  
  useEffect(() => {
    fetchProjects();
  }, []);
  
  const fetchProjects = async () => {
    setIsLoadingProjects(true);
    try {
      const response = await fetch("http://localhost:2024/projects/list");
      if (response.ok) {
        const projects = await response.json();
        setExistingProjects(projects);
      }
    } catch (error) {
      console.error("Failed to fetch projects:", error);
    } finally {
      setIsLoadingProjects(false);
    }
  };
  
  const handleProjectSelect = (value: string) => {
    if (value === "new") {
      onNewProject();
      setSelectedMode("new");
    } else {
      const project = existingProjects.find(p => p.id === value);
      if (project) {
        onSelectProject(project.id, project.name);
        setSelectedMode("existing");
      }
    }
  };
  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FolderOpen className="h-5 w-5" />
          Project Management
        </CardTitle>
        <CardDescription>
          Enter a project name to save your research sessions. Leave empty for one-time queries.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Project Selection */}
        <div className="space-y-2">
          <Label htmlFor="project-select">Select or Create Project</Label>
          <Select value={projectId} onValueChange={handleProjectSelect}>
            <SelectTrigger>
              <SelectValue placeholder="Select a project..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="new">
                <div className="flex items-center gap-2">
                  <FolderOpen className="h-4 w-4" />
                  Create New Project
                </div>
              </SelectItem>
              {existingProjects.length > 0 && (
                <>
                  <div className="px-2 py-1 text-sm font-semibold text-muted-foreground">
                    Existing Projects
                  </div>
                  {existingProjects.map((project) => (
                    <SelectItem key={project.id} value={project.id}>
                      <div className="flex items-center justify-between w-full">
                        <span>{project.name}</span>
                        <span className="text-xs text-muted-foreground ml-2">
                          ({project.session_count} sessions)
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </>
              )}
            </SelectContent>
          </Select>
        </div>
        
        {/* Project Name Input */}
        <div className="space-y-2">
          <Label htmlFor="project-name">
            {selectedMode === "existing" ? "Current Project" : "Project Name"}
          </Label>
          <div className="flex gap-2">
            <Input
              id="project-name"
              value={projectName}
              onChange={(e) => onProjectNameChange(e.target.value)}
              placeholder="Enter project name..."
              className="flex-1"
              disabled={selectedMode === "existing"}
            />
            {selectedMode === "existing" && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  handleProjectSelect("new");
                }}
                title="Create new project"
              >
                <FolderOpen className="h-4 w-4 mr-1" />
                New
              </Button>
            )}
          </div>
          <p className="text-sm text-muted-foreground">
            Project ID: {projectId.slice(0, 8)}...
          </p>
          {projectName.trim() === "" && selectedMode === "new" && (
            <p className="text-sm text-amber-600 dark:text-amber-400 mt-1">
              ⚠️ Prompts will not be saved without a project name
            </p>
          )}
        </div>
        
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="save-prompts" className="flex items-center gap-2">
              <Save className="h-4 w-4" />
              Save Prompts to Markdown
            </Label>
            <p className="text-sm text-muted-foreground">
              Save research sessions as markdown files (requires project name)
            </p>
          </div>
          <Switch
            id="save-prompts"
            checked={savePrompts && projectName.trim() !== ""}
            onCheckedChange={onSavePromptsToggle}
            disabled={projectName.trim() === ""}
          />
        </div>
      </CardContent>
    </Card>
  );
};