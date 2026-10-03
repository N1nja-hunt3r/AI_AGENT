import { createElement, lazy } from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";
import { PageSuspense } from "./index";
import MainLayout from "@/layouts/MainLayout";

const ChatPage = lazy(() => import("@/pages/Chat"));
const SettingsPage = lazy(() => import("@/pages/Settings"));

const wrap = (Page: React.LazyExoticComponent<React.ComponentType>) =>
  createElement(PageSuspense, null, createElement(Page));

const routes = [
  {
    path: "/",
    element: createElement(MainLayout),
    children: [
      { index: true, element: createElement(Navigate, { to: "/chat", replace: true }) },
      { path: "chat", element: wrap(ChatPage) },
      { path: "settings", element: wrap(SettingsPage) },
    ],
  },
  { path: "*", element: createElement(Navigate, { to: "/chat", replace: true }) },
];

export const router = createBrowserRouter(routes, {
  future: { v7_relativeSplatPath: true },
});

export default router;
