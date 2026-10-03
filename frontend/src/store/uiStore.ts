// frontend/src/store/uiStore.ts

import { create } from "zustand";
import { devtools, persist, subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

export type NotificationSeverity = "info" | "success" | "warning" | "error";

export type NotificationPosition =
  | "top-left"
  | "top-center"
  | "top-right"
  | "bottom-left"
  | "bottom-center"
  | "bottom-right";

export interface UINotification {
  id: string;
  severity: NotificationSeverity;
  title: string;
  message?: string;
  duration_ms: number | null;
  is_dismissible: boolean;
  is_dismissed: boolean;
  action?: {
    label: string;
    callback_key: string;
  };
  icon?: string;
  created_at: number;
}

export type DialogId =
  | "settings"
  | "new_conversation"
  | "delete_conversation"
  | "rename_conversation"
  | "memory_detail"
  | "memory_create"
  | "agent_config"
  | "provider_setup"
  | "model_select"
  | "cost_overview"
  | "export_data"
  | "import_data"
  | "keyboard_shortcuts"
  | "confirm_action"
  | "error_detail"
  | (string & NonNullable<unknown>);

export interface DialogState<TProps = Record<string, unknown>> {
  is_open: boolean;
  props: TProps | null;
  opened_at: number | null;
}

export type LoadingOverlayId = string;

export interface LoadingOverlay {
  id: LoadingOverlayId;
  message?: string;
  is_cancellable: boolean;
  progress?: number;
  created_at: number;
}

export type AppPage =
  | "chat"
  | "memory"
  | "agents"
  | "settings"
  | "analytics"
  | "playground"
  | (string & NonNullable<unknown>);

export type SidebarSection =
  | "conversations"
  | "memories"
  | "agents"
  | "tools"
  | "settings";

export interface CommandPaletteState {
  is_open: boolean;
  query: string;
  selected_index: number;
}

export interface UIState {
  sidebar_collapsed: boolean;
  sidebar_width: number;
  sidebar_active_section: SidebarSection;
  mobile_menu_open: boolean;
  current_page: AppPage;
  previous_page: AppPage | null;
  dialogs: Partial<Record<DialogId, DialogState>>;
  notifications: UINotification[];
  notification_position: NotificationPosition;
  max_visible_notifications: number;
  loading_overlays: Record<LoadingOverlayId, LoadingOverlay>;
  global_loading: boolean;
  global_loading_message: string | null;
  command_palette: CommandPaletteState;
  right_panel_open: boolean;
  right_panel_width: number;
  right_panel_content: string | null;
  is_fullscreen: boolean;
  is_compact_mode: boolean;
}

export interface UIActions {
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  setSidebarWidth: (width: number) => void;
  setSidebarActiveSection: (section: SidebarSection) => void;
  setMobileMenuOpen: (open: boolean) => void;
  toggleMobileMenu: () => void;
  navigateTo: (page: AppPage) => void;
  openDialog: (
    id: DialogId,
    props?: Record<string, unknown>
  ) => void;
  closeDialog: (id: DialogId) => void;
  closeAllDialogs: () => void;
  toggleDialog: (id: DialogId, props?: Record<string, unknown>) => void;
  isDialogOpen: (id: DialogId) => boolean;
  getDialogProps: <TProps = Record<string, unknown>>(
    id: DialogId
  ) => TProps | null;
  pushNotification: (
    notification: Omit<
      UINotification,
      "id" | "created_at" | "is_dismissed"
    >
  ) => string;
  dismissNotification: (id: string) => void;
  dismissAllNotifications: () => void;
  removeNotification: (id: string) => void;
  setNotificationPosition: (position: NotificationPosition) => void;
  setMaxVisibleNotifications: (max: number) => void;
  startLoadingOverlay: (
    id: LoadingOverlayId,
    options?: Omit<LoadingOverlay, "id" | "created_at">
  ) => void;
  updateLoadingOverlayProgress: (
    id: LoadingOverlayId,
    progress: number
  ) => void;
  stopLoadingOverlay: (id: LoadingOverlayId) => void;
  stopAllLoadingOverlays: () => void;
  setGlobalLoading: (loading: boolean, message?: string) => void;
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
  setCommandPaletteQuery: (query: string) => void;
  setCommandPaletteSelectedIndex: (index: number) => void;
  setRightPanelOpen: (open: boolean) => void;
  toggleRightPanel: () => void;
  setRightPanelWidth: (width: number) => void;
  setRightPanelContent: (content: string | null) => void;
  setIsFullscreen: (fullscreen: boolean) => void;
  toggleFullscreen: () => void;
  setIsCompactMode: (compact: boolean) => void;
  hasActiveLoadingOverlay: () => boolean;
  getActiveNotifications: () => UINotification[];
}

export type UIStore = UIState & UIActions;

const nanoid = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 11)}`;

const DEFAULT_COMMAND_PALETTE: CommandPaletteState = {
  is_open: false,
  query: "",
  selected_index: 0,
};

export const useUIStore = create<UIStore>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          sidebar_collapsed: false,
          sidebar_width: 280,
          sidebar_active_section: "conversations",
          mobile_menu_open: false,
          current_page: "chat",
          previous_page: null,
          dialogs: {},
          notifications: [],
          notification_position: "top-right",
          max_visible_notifications: 5,
          loading_overlays: {},
          global_loading: false,
          global_loading_message: null,
          command_palette: DEFAULT_COMMAND_PALETTE,
          right_panel_open: false,
          right_panel_width: 380,
          right_panel_content: null,
          is_fullscreen: false,
          is_compact_mode: false,

          setSidebarCollapsed: (collapsed) => {
            set((state) => {
              state.sidebar_collapsed = collapsed;
            });
          },

          toggleSidebar: () => {
            set((state) => {
              state.sidebar_collapsed = !state.sidebar_collapsed;
            });
          },

          setSidebarWidth: (width) => {
            set((state) => {
              state.sidebar_width = Math.max(200, Math.min(600, width));
            });
          },

          setSidebarActiveSection: (section) => {
            set((state) => {
              state.sidebar_active_section = section;
            });
          },

          setMobileMenuOpen: (open) => {
            set((state) => {
              state.mobile_menu_open = open;
            });
          },

          toggleMobileMenu: () => {
            set((state) => {
              state.mobile_menu_open = !state.mobile_menu_open;
            });
          },

          navigateTo: (page) => {
            set((state) => {
              state.previous_page = state.current_page;
              state.current_page = page;
              state.mobile_menu_open = false;
            });
          },

          openDialog: (id, props = {}) => {
            set((state) => {
              state.dialogs[id] = {
                is_open: true,
                props,
                opened_at: Date.now(),
              };
            });
          },

          closeDialog: (id) => {
            set((state) => {
              const dialog = state.dialogs[id];
              if (dialog) {
                dialog.is_open = false;
                dialog.props = null;
                dialog.opened_at = null;
              }
            });
          },

          closeAllDialogs: () => {
            set((state) => {
              for (const id in state.dialogs) {
                const dialog = state.dialogs[id];
                if (dialog) {
                  dialog.is_open = false;
                  dialog.props = null;
                  dialog.opened_at = null;
                }
              }
            });
          },

          toggleDialog: (id, props) => {
            const current = get().dialogs[id];
            if (current?.is_open) {
              get().closeDialog(id);
            } else {
              get().openDialog(id, props);
            }
          },

          isDialogOpen: (id) => {
            return get().dialogs[id]?.is_open ?? false;
          },

          getDialogProps: <TProps = Record<string, unknown>>(
            id: DialogId
          ) => {
            return (get().dialogs[id]?.props ?? null) as TProps | null;
          },

          pushNotification: (notification) => {
            const id = nanoid();
            const newNotification: UINotification = {
              ...notification,
              id,
              is_dismissed: false,
              created_at: Date.now(),
            };

            set((state) => {
              state.notifications.unshift(newNotification);
              if (
                state.notifications.length >
                state.max_visible_notifications * 2
              ) {
                state.notifications = state.notifications.slice(
                  0,
                  state.max_visible_notifications * 2
                );
              }
            });

            if (
              notification.duration_ms !== null &&
              notification.duration_ms > 0
            ) {
              setTimeout(() => {
                get().dismissNotification(id);
              }, notification.duration_ms);
            }

            return id;
          },

          dismissNotification: (id) => {
            set((state) => {
              const n = state.notifications.find((n) => n.id === id);
              if (n) n.is_dismissed = true;
            });
          },

          dismissAllNotifications: () => {
            set((state) => {
              for (const n of state.notifications) {
                n.is_dismissed = true;
              }
            });
          },

          removeNotification: (id) => {
            set((state) => {
              state.notifications = state.notifications.filter(
                (n) => n.id !== id
              );
            });
          },

          setNotificationPosition: (position) => {
            set((state) => {
              state.notification_position = position;
            });
          },

          setMaxVisibleNotifications: (max) => {
            set((state) => {
              state.max_visible_notifications = Math.max(1, max);
            });
          },

          startLoadingOverlay: (id, options = { is_cancellable: false }) => {
            set((state) => {
              state.loading_overlays[id] = {
                id,
                message: options.message,
                is_cancellable: options.is_cancellable,
                progress: options.progress,
                created_at: Date.now(),
              };
            });
          },

          updateLoadingOverlayProgress: (id, progress) => {
            set((state) => {
              const overlay = state.loading_overlays[id];
              if (overlay) {
                overlay.progress = Math.max(0, Math.min(100, progress));
              }
            });
          },

          stopLoadingOverlay: (id) => {
            set((state) => {
              delete state.loading_overlays[id];
            });
          },

          stopAllLoadingOverlays: () => {
            set((state) => {
              state.loading_overlays = {};
            });
          },

          setGlobalLoading: (loading, message) => {
            set((state) => {
              state.global_loading = loading;
              state.global_loading_message = message ?? null;
            });
          },

          openCommandPalette: () => {
            set((state) => {
              state.command_palette.is_open = true;
              state.command_palette.query = "";
              state.command_palette.selected_index = 0;
            });
          },

          closeCommandPalette: () => {
            set((state) => {
              state.command_palette = DEFAULT_COMMAND_PALETTE;
            });
          },

          setCommandPaletteQuery: (query) => {
            set((state) => {
              state.command_palette.query = query;
              state.command_palette.selected_index = 0;
            });
          },

          setCommandPaletteSelectedIndex: (index) => {
            set((state) => {
              state.command_palette.selected_index = index;
            });
          },

          setRightPanelOpen: (open) => {
            set((state) => {
              state.right_panel_open = open;
              if (!open) state.right_panel_content = null;
            });
          },

          toggleRightPanel: () => {
            set((state) => {
              state.right_panel_open = !state.right_panel_open;
              if (!state.right_panel_open) state.right_panel_content = null;
            });
          },

          setRightPanelWidth: (width) => {
            set((state) => {
              state.right_panel_width = Math.max(280, Math.min(800, width));
            });
          },

          setRightPanelContent: (content) => {
            set((state) => {
              state.right_panel_content = content;
              if (content !== null) state.right_panel_open = true;
            });
          },

          setIsFullscreen: (fullscreen) => {
            set((state) => {
              state.is_fullscreen = fullscreen;
            });
          },

          toggleFullscreen: () => {
            set((state) => {
              state.is_fullscreen = !state.is_fullscreen;
            });
          },

          setIsCompactMode: (compact) => {
            set((state) => {
              state.is_compact_mode = compact;
            });
          },

          hasActiveLoadingOverlay: () => {
            return Object.keys(get().loading_overlays).length > 0;
          },

          getActiveNotifications: () => {
            const { notifications, max_visible_notifications } = get();
            return notifications
              .filter((n) => !n.is_dismissed)
              .slice(0, max_visible_notifications);
          },
        }))
      ),
      {
        name: "ai-os-ui",
        partialize: (state) => ({
          sidebar_collapsed: state.sidebar_collapsed,
          sidebar_width: state.sidebar_width,
          sidebar_active_section: state.sidebar_active_section,
          notification_position: state.notification_position,
          max_visible_notifications: state.max_visible_notifications,
          right_panel_width: state.right_panel_width,
          is_compact_mode: state.is_compact_mode,
        }),
      }
    ),
    { name: "UIStore" }
  )
);