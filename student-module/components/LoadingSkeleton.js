/**
 * student-module/components/LoadingSkeleton.js
 * ---------------------------------------------
 * Animated skeleton loading placeholder.
 */
import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet } from 'react-native';
import { Colors, BorderRadius, Spacing } from '../constants/theme';

const SkeletonBlock = ({ width = '100%', height = 16, style, borderRadius = BorderRadius.md }) => {
  const opacity = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, { toValue: 0.7, duration: 800, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 0.3, duration: 800, useNativeDriver: true }),
      ])
    );
    animation.start();
    return () => animation.stop();
  }, []);

  return (
    <Animated.View
      style={[
        { width, height, borderRadius, backgroundColor: '#e5e7eb', opacity },
        style,
      ]}
    />
  );
};

export const DashboardSkeleton = () => (
  <View style={skeletonStyles.container}>
    {/* Profile card */}
    <SkeletonBlock height={100} borderRadius={BorderRadius.xxl} style={{ marginBottom: Spacing.lg }} />
    {/* Stats card */}
    <SkeletonBlock height={120} borderRadius={BorderRadius.xl} style={{ marginBottom: Spacing.lg }} />
    {/* Quick stats row */}
    <View style={skeletonStyles.row}>
      <SkeletonBlock width="48%" height={80} borderRadius={BorderRadius.lg} />
      <SkeletonBlock width="48%" height={80} borderRadius={BorderRadius.lg} />
    </View>
    {/* Subject cards */}
    <SkeletonBlock height={24} width="40%" style={{ marginBottom: Spacing.md, marginTop: Spacing.lg }} />
    <SkeletonBlock height={72} borderRadius={BorderRadius.lg} style={{ marginBottom: Spacing.sm }} />
    <SkeletonBlock height={72} borderRadius={BorderRadius.lg} style={{ marginBottom: Spacing.sm }} />
    <SkeletonBlock height={72} borderRadius={BorderRadius.lg} />
  </View>
);

export const ListSkeleton = ({ count = 5 }) => (
  <View style={skeletonStyles.container}>
    {Array.from({ length: count }).map((_, i) => (
      <SkeletonBlock
        key={i}
        height={72}
        borderRadius={BorderRadius.lg}
        style={{ marginBottom: Spacing.sm }}
      />
    ))}
  </View>
);

export const CardSkeleton = () => (
  <View style={skeletonStyles.card}>
    <SkeletonBlock height={16} width="60%" style={{ marginBottom: Spacing.sm }} />
    <SkeletonBlock height={12} width="40%" style={{ marginBottom: Spacing.md }} />
    <SkeletonBlock height={8} borderRadius={4} />
  </View>
);

const skeletonStyles = StyleSheet.create({
  container: { padding: Spacing.lg },
  row: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: Spacing.lg },
  card: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.xl,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
  },
});

export default SkeletonBlock;
