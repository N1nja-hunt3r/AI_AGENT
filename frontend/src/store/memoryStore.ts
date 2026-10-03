import { create } from "zustand";
import { devtools, subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

// ─── Types ────────────────────────────────────────────────────────────────────

export type MemoryType = "episodic" | "semantic" | "procedural" | "working";

export type MemoryStatus = "active" | "archived" | "deleted";

export interface MemoryTag {
  id: string;
  label: string;
  color?: string;
}

export interface Memory {
  id: string;
  type: MemoryType;
  status: MemoryStatus;
  content: string;
  summary?: string;
  embedding?: number[];
  score?: number; // relevance score from search
  tags: MemoryTag[];
  conversationId?: string;
  agentId?: string;
  metadata?: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
  accessedAt: number;
  accessCount: number;
}

export type SortField = "createdAt" | "updatedAt" | "accessedAt" | "score" | "accessCount";
export type SortOrder = "asc" | "desc";

export interface MemoryFilters {
  types: MemoryType[];
  statuses: MemoryStatus[];
  tags: string[]; // tag IDs
  dateRange: {
    from: number | null;
    to: number | null;
  };
  agentId: string | null;
  conversationId: string | null;
}

export interface MemorySearchState {
  query: string;
  isSearching: boolean;
  results: Memory[];
  totalResults: number;
  page: number;
  pageSize: number;
  sortField: SortField;
  sortOrder: SortOrder;
  error: string | null;
}

export interface MemoryState {
  // Memory storage
  memories: Record<string, Memory>;

  // Available tags
  tags: MemoryTag[];

  // Search & filter state
  search: MemorySearchState;
  filters: MemoryFilters;

  // UI state
  selectedMemoryId: string | null;
  isLoading: boolean;
  error: string | null;
}

export interface MemoryActions {
  // CRUD
  addMemory: (memory: Omit<Memory, "id" | "createdAt" | "updatedAt" | "accessedAt" | "accessCount">) => string;
  updateMemory: (id: string, patch: Partial<Omit<Memory, "id" | "createdAt">>) => void;
  deleteMemory: (id: string) => void;
  archiveMemory: (id: string) => void;
  restoreMemory: (id: string) => void;
  recordAccess: (id: string) => void;
  bulkDeleteMemories: (ids: string[]) => void;
  setMemories: (memories: Memory[]) => void;

  // Tags
  addTag: (tag: Omit<MemoryTag, "id">) => string;
  removeTag: (tagId: string) => void;
  updateTag: (tagId: string, patch: Partial<Omit<MemoryTag, "id">>) => void;

  // Search
  setSearchQuery: (query: string) => void;
  setSearchResults: (results: Memory[], total: number) => void;
  setIsSearching: (isSearching: boolean) => void;
  setSearchPage: (page: number) => void;
  setSearchPageSize: (pageSize: number) => void;
  setSortField: (field: SortField) => void;
  setSortOrder: (order: SortOrder) => void;
  clearSearch: () => void;
  setSearchError: (error: string | null) => void;

  // Filters
  setTypeFilter: (types: MemoryType[]) => void;
  setStatusFilter: (statuses: MemoryStatus[]) => void;
  setTagFilter: (tagIds: string[]) => void;
  setDateRange: (from: number | null, to: number | null) => void;
  setAgentFilter: (agentId: string | null) => void;
  setConversationFilter: (conversationId: string | null) => void;
  clearFilters: () => void;

  // Selection
  selectMemory: (id: string | null) => void;

  // Loading / error
  setIsLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;

  // Derived
  getMemoryById: (id: string) => Memory | null;
  getFilteredMemories: () => Memory[];
}

export type MemoryStore = MemoryState & MemoryActions;

// ─── Defaults ─────────────────────────────────────────────────────────────────

const generateId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

const now = (): number => Date.now();

const DEFAULT_FILTERS: MemoryFilters = {
  types: [],
  statuses: ["active"],
  tags: [],
  dateRange: { from: null, to: null },
  agentId: null,
  conversationId: null,
};

const DEFAULT_SEARCH: MemorySearchState = {
  query: "",
  isSearching: false,
  results: [],
  totalResults: 0,
  page: 1,
  pageSize: 20,
  sortField: "createdAt",
  sortOrder: "desc",
  error: null,
};

// ─── Store ────────────────────────────────────────────────────────────────────

export const useMemoryStore = create<MemoryStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        // ── Initial state ────────────────────────────────────────────────────
        memories: {},
        tags: [],
        search: DEFAULT_SEARCH,
        filters: DEFAULT_FILTERS,
        selectedMemoryId: null,
        isLoading: false,
        error: null,

        // ── CRUD ─────────────────────────────────────────────────────────────
        addMemory: (memory) => {
          const id = generateId();
          const timestamp = now();
          const newMemory: Memory = {
            ...memory,
            id,
            createdAt: timestamp,
            updatedAt: timestamp,
            accessedAt: timestamp,
            accessCount: 0,
          };

          set((state) => {
            state.memories[id] = newMemory;
          });

          return id;
        },

        updateMemory: (id, patch) => {
          set((state) => {
            const memory = state.memories[id];
            if (memory) {
              Object.assign(memory, patch, { updatedAt: now() });
            }
          });
        },

        deleteMemory: (id) => {
          set((state) => {
            state.memories[id] = { ...state.memories[id]!, status: "deleted", updatedAt: now() };
          });
        },

        archiveMemory: (id) => {
          set((state) => {
            const memory = state.memories[id];
            if (memory) {
              memory.status = "archived";
              memory.updatedAt = now();
            }
          });
        },

        restoreMemory: (id) => {
          set((state) => {
            const memory = state.memories[id];
            if (memory) {
              memory.status = "active";
              memory.updatedAt = now();
            }
          });
        },

        recordAccess: (id) => {
          set((state) => {
            const memory = state.memories[id];
            if (memory) {
              memory.accessedAt = now();
              memory.accessCount += 1;
            }
          });
        },

        bulkDeleteMemories: (ids) => {
          set((state) => {
            const timestamp = now();
            ids.forEach((id) => {
              if (state.memories[id]) {
                state.memories[id]!.status = "deleted";
                state.memories[id]!.updatedAt = timestamp;
              }
            });
          });
        },

        setMemories: (memories) => {
          set((state) => {
            state.memories = Object.fromEntries(memories.map((m) => [m.id, m]));
          });
        },

        // ── Tags ──────────────────────────────────────────────────────────────
        addTag: (tag) => {
          const id = generateId();
          set((state) => {
            state.tags.push({ ...tag, id });
          });
          return id;
        },

        removeTag: (tagId) => {
          set((state) => {
            state.tags = state.tags.filter((t) => t.id !== tagId);
            // Remove tag from all memories
            Object.values(state.memories).forEach((memory) => {
              memory.tags = memory.tags.filter((t) => t.id !== tagId);
            });
          });
        },

        updateTag: (tagId, patch) => {
          set((state) => {
            const tag = state.tags.find((t) => t.id === tagId);
            if (tag) Object.assign(tag, patch);
          });
        },

        // ── Search ────────────────────────────────────────────────────────────
        setSearchQuery: (query) => {
          set((state) => {
            state.search.query = query;
            state.search.page = 1;
          });
        },

        setSearchResults: (results, total) => {
          set((state) => {
            state.search.results = results;
            state.search.totalResults = total;
            state.search.isSearching = false;
            state.search.error = null;
          });
        },

        setIsSearching: (isSearching) => {
          set((state) => {
            state.search.isSearching = isSearching;
          });
        },

        setSearchPage: (page) => {
          set((state) => {
            state.search.page = page;
          });
        },

        setSearchPageSize: (pageSize) => {
          set((state) => {
            state.search.pageSize = pageSize;
            state.search.page = 1;
          });
        },

        setSortField: (field) => {
          set((state) => {
            state.search.sortField = field;
            state.search.page = 1;
          });
        },

        setSortOrder: (order) => {
          set((state) => {
            state.search.sortOrder = order;
          });
        },

        clearSearch: () => {
          set((state) => {
            state.search = DEFAULT_SEARCH;
          });
        },

        setSearchError: (error) => {
          set((state) => {
            state.search.error = error;
            state.search.isSearching = false;
          });
        },

        // ── Filters ───────────────────────────────────────────────────────────
        setTypeFilter: (types) => {
          set((state) => {
            state.filters.types = types;
          });
        },

        setStatusFilter: (statuses) => {
          set((state) => {
            state.filters.statuses = statuses;
          });
        },

        setTagFilter: (tagIds) => {
          set((state) => {
            state.filters.tags = tagIds;
          });
        },

        setDateRange: (from, to) => {
          set((state) => {
            state.filters.dateRange = { from, to };
          });
        },

        setAgentFilter: (agentId) => {
          set((state) => {
            state.filters.agentId = agentId;
          });
        },

        setConversationFilter: (conversationId) => {
          set((state) => {
            state.filters.conversationId = conversationId;
          });
        },

        clearFilters: () => {
          set((state) => {
            state.filters = DEFAULT_FILTERS;
          });
        },

        // ── Selection ─────────────────────────────────────────────────────────
        selectMemory: (id) => {
          set((state) => {
            state.selectedMemoryId = id;
          });
          if (id) get().recordAccess(id);
        },

        // ── Loading / error ───────────────────────────────────────────────────
        setIsLoading: (isLoading) => {
          set((state) => {
            state.isLoading = isLoading;
          });
        },

        setError: (error) => {
          set((state) => {
            state.error = error;
          });
        },

        // ── Derived ───────────────────────────────────────────────────────────
        getMemoryById: (id) => get().memories[id] ?? null,

        getFilteredMemories: () => {
          const { memories, filters } = get();
          return Object.values(memories).filter((memory) => {
            if (memory.status === "deleted") return false;
            if (filters.types.length > 0 && !filters.types.includes(memory.type)) return false;
            if (filters.statuses.length > 0 && !filters.statuses.includes(memory.status)) return false;
            if (filters.tags.length > 0 && !filters.tags.some((tid) => memory.tags.some((t) => t.id === tid))) return false;
            if (filters.agentId && memory.agentId !== filters.agentId) return false;
            if (filters.conversationId && memory.conversationId !== filters.conversationId) return false;
            if (filters.dateRange.from && memory.createdAt < filters.dateRange.from) return false;
            if (filters.dateRange.to && memory.createdAt > filters.dateRange.to) return false;
            return true;
          });
        },
      }))
    ),
    { name: "MemoryStore" }
  )
);