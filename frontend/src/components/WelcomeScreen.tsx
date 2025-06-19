import { InputForm } from "./InputForm";
import { ProjectSettings } from "./ProjectSettings";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string,
    model: string
  ) => void;
  onCancel: () => void;
  isLoading: boolean;
  projectName?: string;
  projectId?: string;
  savePrompts?: boolean;
  onProjectNameChange?: (name: string) => void;
  onSavePromptsToggle?: (enabled: boolean) => void;
  onNewProject?: () => void;
  onSelectProject?: (projectId: string, projectName: string) => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
  projectName,
  projectId,
  savePrompts,
  onProjectNameChange,
  onSavePromptsToggle,
  onNewProject,
  onSelectProject,
}) => (
  <div className="h-full flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-4">
    <div>
      <h1 className="text-5xl md:text-6xl font-semibold text-neutral-100 mb-3">
        K-RAG Agent
      </h1>
      <p className="text-xl md:text-2xl text-neutral-400">
        How can I help you today?
      </p>
    </div>
    <div className="w-full max-w-xl">
      <ProjectSettings
        projectName={projectName || ""}
        projectId={projectId || ""}
        savePrompts={savePrompts || false}
        onProjectNameChange={onProjectNameChange || (() => {})}
        onSavePromptsToggle={onSavePromptsToggle || (() => {})}
        onNewProject={onNewProject || (() => {})}
        onSelectProject={onSelectProject || (() => {})}
      />
    </div>
    <div className="w-full mt-4">
      <InputForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={false}
      />
    </div>
    <p className="text-xs text-neutral-500">
      Powered by Google Gemini and LangChain LangGraph.
    </p>
  </div>
);
