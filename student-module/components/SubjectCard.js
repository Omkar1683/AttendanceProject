/**
 * student-module/components/SubjectCard.js
 * -----------------------------------------
 * Card displaying subject-wise attendance with progress bar and status badge.
 */
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import ProgressBar from './ProgressBar';
import StatusBadge from './StatusBadge';

const SubjectCard = ({ subject, onPress, style }) => {
  const { name, percentage, present, total_classes, absent, status } = subject;

  const Wrapper = onPress ? TouchableOpacity : View;

  return (
    <Wrapper
      style={[cardStyles.container, style]}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={cardStyles.topRow}>
        <View style={cardStyles.info}>
          <Text style={cardStyles.name} numberOfLines={1}>{name}</Text>
          <Text style={cardStyles.detail}>
            {present}/{total_classes} classes • {absent || (total_classes - present)} absent
          </Text>
        </View>
        <View style={cardStyles.right}>
          <Text style={[
            cardStyles.percentage,
            { color: percentage >= 75 ? Colors.success : Colors.danger },
          ]}>
            {percentage}%
          </Text>
          <StatusBadge percentage={percentage} />
        </View>
      </View>
      <ProgressBar percentage={percentage} style={cardStyles.progress} />
    </Wrapper>
  );
};

const cardStyles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg,
    padding: Spacing.lg,
    marginBottom: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.borderLight,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.md,
  },
  info: { flex: 1, marginRight: Spacing.md },
  name: {
    fontSize: FontSizes.md,
    fontWeight: 'bold',
    color: Colors.text,
    marginBottom: 2,
  },
  detail: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
  },
  right: { alignItems: 'flex-end' },
  percentage: {
    fontSize: FontSizes.lg,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  progress: { marginTop: 0 },
});

export default SubjectCard;
