import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Button } from "@/components/ui/button";

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [waitingForRefinement, setWaitingForRefinement] = useState(false);
  const [refinementOptions, setRefinementOptions] = useState<string[]>([]);
  
  // Project management state
  const [projectName, setProjectName] = useState("");
  const [projectId, setProjectId] = useState(
    () => `proj_${Date.now().toString(36)}_${Math.random().toString(36).substr(2, 9)}`
  );
  const [savePrompts, setSavePrompts] = useState(true);
  const thread = useStream({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "k-rag-agent",
    messagesKey: "messages",
    onUpdateEvent: (event: Record<string, any>) => {
      let processedEvent: ProcessedEvent | null = null;
      if (event.refine_query) {
        processedEvent = {
          title: "Query Refinement",
          data: event.refine_query?.needs_refinement 
            ? "Analyzing query for clarity" 
            : "Query is clear, proceeding",
        };
        if (event.refine_query?.needs_refinement && event.refine_query?.refinement_suggestions) {
          setRefinementOptions(event.refine_query.refinement_suggestions);
        }
      } else if (event.wait_for_user) {
        setWaitingForRefinement(true);
        processedEvent = {
          title: "Awaiting User Input",
          data: "Waiting for query clarification",
        };
      } else if (event.generate_query) {
        setWaitingForRefinement(false);
        processedEvent = {
          title: "Generating Search Queries",
          data: event.generate_query?.search_query?.join(", ") || "",
        };
      } else if (event.web_research) {
        const sources = event.web_research.sources_gathered || [];
        const numSources = sources.length;
        const uniqueLabels = [
          ...new Set(sources.map((s: { label?: string }) => s.label).filter(Boolean)),
        ];
        const exampleLabels = uniqueLabels.slice(0, 3).join(", ");
        processedEvent = {
          title: "Web Research",
          data: `Gathered ${numSources} sources. Related to: ${
            exampleLabels || "N/A"
          }.`,
        };
      } else if (event.reflection) {
        processedEvent = {
          title: "Reflection",
          data: "Analysing Web Research Results",
        };
      } else if (event.finalize_answer) {
        processedEvent = {
          title: "Finalizing Answer",
          data: "Composing and presenting the final answer.",
        };
        hasFinalizeEventOccurredRef.current = true;
      }
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
    onError: (error: unknown) => {
      setError(error instanceof Error ? error.message : String(error));
    },
  });

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      
      // Check if we're in refinement mode
      if (waitingForRefinement) {
        // Handle refinement response
        const currentState = {};
        let updatedMessages = thread.messages || [];
        
        // User's response to the clarifying question
        updatedMessages.push({
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        });
        
        // Add user response to refinement conversation
        const refinementConv = [];
        refinementConv.push({ role: "human", content: submittedInputValue });
        
        // Check if user wants to proceed with search
        const proceedPhrases = ["let's go", "lets go", "go ahead", "proceed", "search now", "that's enough", "start searching", "go", "ready"];
        const userWantsToSearch = proceedPhrases.some(phrase => 
          submittedInputValue.toLowerCase().includes(phrase)
        );
        
        thread.submit({
          ...currentState,
          messages: updatedMessages,
          user_ready_to_search: userWantsToSearch,
          needs_refinement: false,
          refinement_conversation: refinementConv,
        });
        
        // Don't clear refinement state yet - let backend decide if more refinement is needed
        return;
      }
      
      // Normal submission (not in refinement mode)
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      // convert effort to, initial_search_query_count and max_research_loops
      // low means max 1 loop and 1 query
      // medium means max 3 loops and 3 queries
      // high means max 10 loops and 5 queries
      let initial_search_query_count = 0;
      let max_research_loops = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          break;
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        reasoning_model: model,
        needs_refinement: false,
        user_approved_refinement: false,
        refinement_suggestions: [],
        original_query: submittedInputValue,
        refinement_conversation: [],
        user_ready_to_search: false,
        // Project management fields
        project_id: projectId,
        project_name: projectName,
        save_prompts: savePrompts,
        saved_prompt_paths: [],
      });
    },
    [thread, waitingForRefinement, refinementOptions, projectId, projectName, savePrompts]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
          {thread.messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
              projectName={projectName}
              projectId={projectId}
              savePrompts={savePrompts}
              onProjectNameChange={setProjectName}
              onSavePromptsToggle={setSavePrompts}
              onNewProject={() => {
                setProjectId(`proj_${Date.now().toString(36)}_${Math.random().toString(36).substr(2, 9)}`);
                setProjectName("New Project");
              }}
            />
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="flex flex-col items-center justify-center gap-4">
                <h1 className="text-2xl text-red-400 font-bold">Error</h1>
                <p className="text-red-400">{JSON.stringify(error)}</p>

                <Button
                  variant="destructive"
                  onClick={() => window.location.reload()}
                >
                  Retry
                </Button>
              </div>
            </div>
          ) : (
            <ChatMessagesView
              messages={thread.messages}
              isLoading={thread.isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
              waitingForRefinement={waitingForRefinement}
            />
          )}
      </main>
    </div>
  );
}
