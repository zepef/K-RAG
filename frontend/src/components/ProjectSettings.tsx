import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import { Button } from "./ui/button";
import { FolderOpen, Save } from "lucide-react";

interface ProjectSettingsProps {
  projectName: string;
  projectId: string;
  savePrompts: boolean;
  onProjectNameChange: (name: string) => void;
  onSavePromptsToggle: (enabled: boolean) => void;
  onNewProject: () => void;
}

export const ProjectSettings: React.FC<ProjectSettingsProps> = ({
  projectName,
  projectId,
  savePrompts,
  onProjectNameChange,
  onSavePromptsToggle,
  onNewProject,
}) => {
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
        <div className="space-y-2">
          <Label htmlFor="project-name">Project Name</Label>
          <div className="flex gap-2">
            <Input
              id="project-name"
              value={projectName}
              onChange={(e) => onProjectNameChange(e.target.value)}
              placeholder="Enter project name (optional)..."
              className="flex-1"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={onNewProject}
              title="Create new project"
            >
              New
            </Button>
          </div>
          <p className="text-sm text-muted-foreground">
            Project ID: {projectId.slice(0, 8)}...
          </p>
          {projectName.trim() === "" && (
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