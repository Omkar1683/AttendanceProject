/**
 * student-module/components/EmptyState.js
 * ----------------------------------------
 * Reusable empty state component with icon, title, and subtitle.
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';

const EmptyState = ({ icon = '📭', title = 'No Data', subtitle = '', style }) => (
  <View style={[emptyStyles.container, style]}>
    <Text style={emptyStyles.icon}>{icon}</Text>
    <Text style={emptyStyles.title}>{title}</Text>
    {!!subtitle && <Text style={emptyStyles.subtitle}>{subtitle}</Text>}
  </View>
);

const emptyStyles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 48,
    paddingHorizontal: Spacing.xxl,
  },
  icon: { fontSize: 48, marginBottom: Spacing.lg },
  title: {
    fontSize: FontSizes.lg,
    fontWeight: '600',
    color: Colors.text,
    textAlign: 'center',
    marginBottom: Spacing.sm,
  },
  subtitle: {
    fontSize: FontSizes.md,
    color: Colors.textTertiary,
    textAlign: 'center',
    lineHeight: 20,
  },
});

export default EmptyState;
