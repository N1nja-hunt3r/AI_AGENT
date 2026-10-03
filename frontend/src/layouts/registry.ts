export { default as MainLayout } from "./MainLayout";
export { default as Sidebar } from "./Sidebar";
export { default as Navbar } from "./Navbar";

export type { } from "./MainLayout";

import MainLayout from "./MainLayout";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";

export interface LayoutRegistry {
  MainLayout: typeof MainLayout;
  Sidebar: typeof Sidebar;
  Navbar: typeof Navbar;
}

const registry: LayoutRegistry = {
  MainLayout,
  Sidebar,
  Navbar,
};

export default registry;
