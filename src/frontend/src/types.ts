/** A single source/chunk returned from the search API. */
export interface Source {
  tipo?: string;
  numero?: string;
  ano?: number;
  orgao?: string;
  status?: string;
  score?: number;
  chunk_text?: string;
}

/** A chat message exchanged between user and assistant. */
export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

/** Model info returned by /api/model-info. */
export interface ModelInfo {
  label: string;
  provider: string;
}

/** Request body sent to POST /search. */
export interface SearchPayload {
  query: string;
  filter_status?: string;
  filter_tipo?: string;
  filter_ano?: number;
  google_api_key?: string;
  selected_model?: string;
}

/** Response from POST /upload. */
export interface UploadResponse {
  status: string;
  filename: string;
  metadata: {
    tipo?: string;
    numero?: string;
    ano?: number;
    orgao?: string;
    status?: string;
    assunto_resumo?: string;
    tags?: string[];
  };
  chunks_created: number;
}

/** Props for the Sidebar component. */
export interface SidebarProps {
  filterStatus: string;
  setFilterStatus: (v: string) => void;
  filterTipo: string;
  setFilterTipo: (v: string) => void;
  filterAno: string;
  setFilterAno: (v: string) => void;
  selectedModel: string;
  setSelectedModel: (v: string) => void;
  uploadApiKey: string;
  setUploadApiKey: (v: string) => void;
  onClearChat: () => void;
  onUploadFile: (file: File) => Promise<void>;
  isUploading: boolean;
  isOpen: boolean;
  onClose: () => void;
}

/** Props for the ChatMessage component. */
export interface ChatMessageProps {
  message: Message;
}
