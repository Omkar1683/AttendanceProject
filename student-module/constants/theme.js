/**
 * student-module/constants/theme.js
 * ---------------------------------
 * Shared theme constants for the Student Module.
 * Reuses the existing AttendAI color palette from the Teacher Module
 * to maintain visual consistency.
 */

export const Colors = {
  // Primary brand
  primary: '#2563eb',
  primaryLight: '#eff6ff',
  primaryDark: '#1d4ed8',
  primaryBorder: '#bfdbfe',

  // Accent / Indigo (used for student profile card)
  accent: '#4f46e5',
  accentLight: '#eef2ff',

  // Success / Safe
  success: '#16a34a',
  successLight: '#dcfce7',
  successBg: '#f0fdf4',

  // Danger / Warning
  danger: '#dc2626',
  dangerLight: '#fee2e2',
  dangerBg: '#fef2f2',

  // Warning / Amber
  warning: '#f59e0b',
  warningLight: '#fef3c7',
  warningBg: '#fffbeb',

  // Orange
  orange: '#ea580c',
  orangeLight: '#ffedd5',

  // Neutrals
  background: '#f9fafb',
  surface: '#ffffff',
  surfaceAlt: '#f3f4f6',
  border: '#e5e7eb',
  borderLight: '#f3f4f6',

  // Text
  text: '#1f2937',
  textSecondary: '#4b5563',
  textTertiary: '#6b7280',
  textMuted: '#9ca3af',
  textWhite: '#ffffff',

  // Shadows
  shadow: 'rgba(0,0,0,0.08)',
};

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,
};

export const FontSizes = {
  xs: 10,
  sm: 12,
  md: 14,
  lg: 16,
  xl: 18,
  xxl: 20,
  xxxl: 24,
  title: 28,
};

export const BorderRadius = {
  sm: 6,
  md: 8,
  lg: 12,
  xl: 16,
  xxl: 20,
  full: 999,
};
