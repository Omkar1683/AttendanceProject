/**
 * student-module/components/ErrorState.js
 * ----------------------------------------
 * Error state with retry button.
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';

const ErrorState = ({ message = 'Something went wrong', onRetry, style }) => (
  <View style={[errorStyles.container, style]}>
    <Text style={errorStyles.icon}>⚠️</Text>
    <Text style={errorStyles.message}>{message}</Text>
    {onRetry && (
      <TouchableOpacity style={errorStyles.retryButton} onPress={onRetry} activeOpacity={0.7}>
        <Text style={errorStyles.retryText}>Retry</Text>
      </TouchableOpacity>
    )}
  </View>
);

const errorStyles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 48,
    paddingHorizontal: Spacing.xxl,
  },
  icon: { fontSize: 48, marginBottom: Spacing.lg },
  message: {
    fontSize: FontSizes.md,
    color: Colors.textTertiary,
    textAlign: 'center',
    marginBottom: Spacing.xl,
    lineHeight: 20,
  },
  retryButton: {
    backgroundColor: Colors.primary,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xxxl,
    borderRadius: BorderRadius.lg,
  },
  retryText: {
    color: Colors.textWhite,
    fontWeight: '600',
    fontSize: FontSizes.md,
  },
});

export default ErrorState;
