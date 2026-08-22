/**
 * student-module/components/ProgressBar.js
 * -----------------------------------------
 * Animated attendance progress bar.
 */
import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet } from 'react-native';
import { Colors, BorderRadius } from '../constants/theme';

const ProgressBar = ({
  percentage = 0,
  height = 8,
  color,
  backgroundColor = Colors.surfaceAlt,
  borderRadius = 4,
  animated = true,
  style,
}) => {
  const animValue = useRef(new Animated.Value(0)).current;
  const barColor = color || (percentage >= 75 ? Colors.success : percentage >= 60 ? Colors.warning : Colors.danger);

  useEffect(() => {
    if (animated) {
      Animated.timing(animValue, {
        toValue: Math.min(percentage, 100),
        duration: 800,
        useNativeDriver: false,
      }).start();
    } else {
      animValue.setValue(Math.min(percentage, 100));
    }
  }, [percentage]);

  const width = animValue.interpolate({
    inputRange: [0, 100],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={[{ height, backgroundColor, borderRadius, overflow: 'hidden' }, style]}>
      <Animated.View
        style={{
          height: '100%',
          width,
          backgroundColor: barColor,
          borderRadius,
        }}
      />
    </View>
  );
};

export default ProgressBar;
