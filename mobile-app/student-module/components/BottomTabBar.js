/**
 * student-module/components/BottomTabBar.js
 * -------------------------------------------
 * Bottom tab navigation bar for the Student Module.
 * Matches the Teacher Module's visual style.
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing } from '../constants/theme';

const TABS = [
  { key: 'Dashboard',     icon: 'home-outline',          iconActive: 'home',          label: 'Home' },
  { key: 'Attendance',    icon: 'calendar-outline',      iconActive: 'calendar',      label: 'Attendance' },
  { key: 'Analytics',     icon: 'bar-chart-outline',     iconActive: 'bar-chart',     label: 'Analytics' },
  { key: 'Notifications', icon: 'notifications-outline', iconActive: 'notifications', label: 'Alerts' },
  { key: 'Profile',       icon: 'person-outline',        iconActive: 'person',        label: 'Profile' },
];

const BottomTabBar = ({ activeTab, onTabPress, unreadCount = 0 }) => {
  return (
    <View style={tabStyles.container}>
      {TABS.map(tab => {
        const isActive = activeTab === tab.key;
        return (
          <TouchableOpacity
            key={tab.key}
            style={tabStyles.tab}
            onPress={() => onTabPress(tab.key)}
            activeOpacity={0.7}
          >
            <View>
              <Ionicons
                name={isActive ? tab.iconActive : tab.icon}
                size={22}
                color={isActive ? Colors.primary : Colors.textMuted}
              />
              {tab.key === 'Notifications' && unreadCount > 0 && (
                <View style={tabStyles.badge}>
                  <Text style={tabStyles.badgeText}>
                    {unreadCount > 9 ? '9+' : unreadCount}
                  </Text>
                </View>
              )}
            </View>
            <Text style={[
              tabStyles.label,
              isActive && tabStyles.labelActive,
            ]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const tabStyles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 8,
    paddingBottom: 12,
    backgroundColor: Colors.surface,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  tab: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 4,
    minWidth: 56,
  },
  label: {
    fontSize: 10,
    marginTop: 3,
    color: Colors.textMuted,
    fontWeight: '500',
  },
  labelActive: {
    color: Colors.primary,
    fontWeight: '600',
  },
  badge: {
    position: 'absolute',
    top: -4,
    right: -8,
    backgroundColor: Colors.danger,
    borderRadius: 8,
    minWidth: 16,
    height: 16,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 3,
  },
  badgeText: {
    color: Colors.textWhite,
    fontSize: 9,
    fontWeight: 'bold',
  },
});

export default BottomTabBar;
