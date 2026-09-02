/**
 * student-module/components/StatusBadge.js
 * -----------------------------------------
 * Attendance status badge (Safe / Warning / Defaulter).
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { getAttendanceStatus } from '../utils/attendanceCalculations';

const StatusBadge = ({ percentage, size = 'sm', style }) => {
  const status = getAttendanceStatus(percentage);

  const colorMap = {
    success: { bg: Colors.successLight, text: Colors.success },
    warning: { bg: Colors.warningLight, text: Colors.warning },
    danger: { bg: Colors.dangerLight, text: Colors.danger },
  };
  const colors = colorMap[status.color] || colorMap.success;

  const isLarge = size === 'lg';

  return (
    <View style={[
      badgeStyles.badge,
      { backgroundColor: colors.bg },
      isLarge && badgeStyles.badgeLarge,
      style,
    ]}>
      <Text style={[
        badgeStyles.text,
        { color: colors.text },
        isLarge && badgeStyles.textLarge,
      ]}>
        {status.label}
      </Text>
    </View>
  );
};

const badgeStyles = StyleSheet.create({
  badge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: BorderRadius.sm,
    alignSelf: 'flex-start',
  },
  badgeLarge: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
  },
  text: {
    fontSize: FontSizes.xs,
    fontWeight: 'bold',
  },
  textLarge: {
    fontSize: FontSizes.sm,
  },
});

export default StatusBadge;
