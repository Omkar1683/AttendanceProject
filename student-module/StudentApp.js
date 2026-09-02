/**
 * student-module/StudentApp.js
 * ------------------------------
 * Main entry point for the Student Module.
 * Provides tab-based navigation between all student screens.
 * This component is imported by mobile-app/App.js and rendered
 * when a user with role 'student' is logged in.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from './constants/theme';
import BottomTabBar from './components/BottomTabBar';
import studentApi from './services/studentApi';

// Screens
import StudentDashboard from './screens/StudentDashboard';
import SubjectAttendance from './screens/SubjectAttendance';
import AttendanceCalendar from './screens/AttendanceCalendar';
import AttendanceHistory from './screens/AttendanceHistory';
import StudentNotifications from './screens/StudentNotifications';
import AttendanceAnalytics from './screens/AttendanceAnalytics';
import AttendancePrediction from './screens/AttendancePrediction';
import StudentProfile from './screens/StudentProfile';
import ExportAttendance from './screens/ExportAttendance';

// Tab → Default Screen mapping
const TAB_SCREENS = {
  Dashboard: 'StudentDashboard',
  Attendance: 'SubjectAttendance',
  Analytics: 'AttendanceAnalytics',
  Notifications: 'StudentNotifications',
  Profile: 'StudentProfile',
};

// Screen titles for the header
const SCREEN_TITLES = {
  StudentDashboard: 'Dashboard',
  SubjectAttendance: 'Subjects',
  AttendanceCalendar: 'Calendar',
  AttendanceHistory: 'History',
  StudentNotifications: 'Notifications',
  AttendanceAnalytics: 'Analytics',
  AttendancePrediction: 'Prediction',
  StudentProfile: 'Profile',
  ExportAttendance: 'Export',
};

// Screens that have a "back" button (sub-screens within a tab)
const BACK_SCREENS = {
  AttendanceCalendar: 'SubjectAttendance',
  AttendanceHistory: 'SubjectAttendance',
  AttendancePrediction: 'AttendanceAnalytics',
  ExportAttendance: 'StudentProfile',
};

// Which tab a screen belongs to
const SCREEN_TAB = {
  StudentDashboard: 'Dashboard',
  SubjectAttendance: 'Attendance',
  AttendanceCalendar: 'Attendance',
  AttendanceHistory: 'Attendance',
  StudentNotifications: 'Notifications',
  AttendanceAnalytics: 'Analytics',
  AttendancePrediction: 'Analytics',
  StudentProfile: 'Profile',
  ExportAttendance: 'Profile',
};

const StudentApp = ({ userInfo, onLogout }) => {
  const [currentScreen, setCurrentScreen] = useState('StudentDashboard');
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [unreadNotifications, setUnreadNotifications] = useState(0);
  const socketRef = useRef(null);

  // Navigate to a specific screen
  const navigateTo = useCallback((screen) => {
    setCurrentScreen(screen);
    const tab = SCREEN_TAB[screen];
    if (tab) setActiveTab(tab);
  }, []);

  // Handle tab press
  const handleTabPress = useCallback((tab) => {
    setActiveTab(tab);
    setCurrentScreen(TAB_SCREENS[tab]);
  }, []);

  // Handle back navigation
  const handleBack = useCallback(() => {
    const backTo = BACK_SCREENS[currentScreen];
    if (backTo) {
      navigateTo(backTo);
    }
  }, [currentScreen, navigateTo]);

  // Socket.IO connection for real-time updates
  useEffect(() => {
    if (userInfo?.id) {
      const socket = studentApi.connectStudentSocket(userInfo.id, (data) => {
        // When attendance is marked, we could show a toast or refresh data
        console.log('[StudentApp] Real-time attendance update:', data);
      });
      socketRef.current = socket;
      return () => socket.close();
    }
  }, [userInfo?.id]);

  // Load initial notification count
  useEffect(() => {
    const loadUnread = async () => {
      try {
        const res = await studentApi.getNotifications(1, 1);
        if (res.status === 'success') {
          setUnreadNotifications(res.data?.unread || 0);
        }
      } catch {}
    };
    loadUnread();
  }, []);

  const showBackButton = !!BACK_SCREENS[currentScreen];
  const screenTitle = SCREEN_TITLES[currentScreen] || 'Student';

  // Render current screen
  const renderScreen = () => {
    switch (currentScreen) {
      case 'StudentDashboard':
        return <StudentDashboard userInfo={userInfo} navigateTo={navigateTo} />;
      case 'SubjectAttendance':
        return <SubjectAttendance navigateTo={navigateTo} />;
      case 'AttendanceCalendar':
        return <AttendanceCalendar navigateTo={navigateTo} />;
      case 'AttendanceHistory':
        return <AttendanceHistory navigateTo={navigateTo} />;
      case 'StudentNotifications':
        return (
          <StudentNotifications
            onUnreadChange={setUnreadNotifications}
          />
        );
      case 'AttendanceAnalytics':
        return <AttendanceAnalytics navigateTo={navigateTo} />;
      case 'AttendancePrediction':
        return <AttendancePrediction />;
      case 'StudentProfile':
        return (
          <StudentProfile
            userInfo={userInfo}
            onLogout={onLogout}
            navigateTo={navigateTo}
          />
        );
      case 'ExportAttendance':
        return <ExportAttendance navigateTo={navigateTo} />;
      default:
        return <StudentDashboard userInfo={userInfo} navigateTo={navigateTo} />;
    }
  };

  return (
    <View style={appStyles.container}>
      <StatusBar barStyle="dark-content" backgroundColor={Colors.surface} />

      {/* ── Header ───────────────────────────────────────────────────── */}
      <View style={appStyles.header}>
        {showBackButton ? (
          <TouchableOpacity onPress={handleBack} style={appStyles.headerBtn}>
            <Ionicons name="chevron-back" size={24} color={Colors.text} />
          </TouchableOpacity>
        ) : (
          <View style={appStyles.headerBtn} />
        )}
        <Text style={appStyles.headerTitle}>{screenTitle}</Text>
        <View style={appStyles.headerBtn} />
      </View>

      {/* ── Sub-navigation for Attendance tab ────────────────────────── */}
      {activeTab === 'Attendance' && !BACK_SCREENS[currentScreen] && (
        <View style={appStyles.subNav}>
          <TouchableOpacity
            style={[appStyles.subNavBtn, currentScreen === 'SubjectAttendance' && appStyles.subNavBtnActive]}
            onPress={() => navigateTo('SubjectAttendance')}
          >
            <Ionicons name="book-outline" size={14} color={currentScreen === 'SubjectAttendance' ? Colors.primary : Colors.textMuted} />
            <Text style={[appStyles.subNavText, currentScreen === 'SubjectAttendance' && appStyles.subNavTextActive]}>Subjects</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[appStyles.subNavBtn, currentScreen === 'AttendanceCalendar' && appStyles.subNavBtnActive]}
            onPress={() => navigateTo('AttendanceCalendar')}
          >
            <Ionicons name="calendar-outline" size={14} color={currentScreen === 'AttendanceCalendar' ? Colors.primary : Colors.textMuted} />
            <Text style={[appStyles.subNavText, currentScreen === 'AttendanceCalendar' && appStyles.subNavTextActive]}>Calendar</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[appStyles.subNavBtn, currentScreen === 'AttendanceHistory' && appStyles.subNavBtnActive]}
            onPress={() => navigateTo('AttendanceHistory')}
          >
            <Ionicons name="time-outline" size={14} color={currentScreen === 'AttendanceHistory' ? Colors.primary : Colors.textMuted} />
            <Text style={[appStyles.subNavText, currentScreen === 'AttendanceHistory' && appStyles.subNavTextActive]}>History</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* ── Sub-navigation for Analytics tab ─────────────────────────── */}
      {activeTab === 'Analytics' && !BACK_SCREENS[currentScreen] && (
        <View style={appStyles.subNav}>
          <TouchableOpacity
            style={[appStyles.subNavBtn, currentScreen === 'AttendanceAnalytics' && appStyles.subNavBtnActive]}
            onPress={() => navigateTo('AttendanceAnalytics')}
          >
            <Ionicons name="bar-chart-outline" size={14} color={currentScreen === 'AttendanceAnalytics' ? Colors.primary : Colors.textMuted} />
            <Text style={[appStyles.subNavText, currentScreen === 'AttendanceAnalytics' && appStyles.subNavTextActive]}>Charts</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[appStyles.subNavBtn, currentScreen === 'AttendancePrediction' && appStyles.subNavBtnActive]}
            onPress={() => navigateTo('AttendancePrediction')}
          >
            <Ionicons name="trending-up-outline" size={14} color={currentScreen === 'AttendancePrediction' ? Colors.primary : Colors.textMuted} />
            <Text style={[appStyles.subNavText, currentScreen === 'AttendancePrediction' && appStyles.subNavTextActive]}>Prediction</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* ── Screen Content ───────────────────────────────────────────── */}
      <View style={appStyles.screenContainer}>
        {renderScreen()}
      </View>

      {/* ── Bottom Tab Bar ───────────────────────────────────────────── */}
      <BottomTabBar
        activeTab={activeTab}
        onTabPress={handleTabPress}
        unreadCount={unreadNotifications}
      />
    </View>
  );
};

const appStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  // Header
  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: Spacing.lg, paddingVertical: Spacing.md,
    backgroundColor: Colors.surface, borderBottomWidth: 1, borderBottomColor: Colors.border,
  },
  headerTitle: { fontSize: FontSizes.xl, fontWeight: 'bold', color: Colors.text },
  headerBtn: { width: 32, alignItems: 'center' },
  // Sub Navigation
  subNav: {
    flexDirection: 'row', backgroundColor: Colors.surface,
    paddingHorizontal: Spacing.lg, paddingBottom: Spacing.sm,
    borderBottomWidth: 1, borderBottomColor: Colors.border,
    gap: Spacing.xs,
  },
  subNavBtn: {
    flexDirection: 'row', alignItems: 'center',
    paddingVertical: Spacing.sm, paddingHorizontal: Spacing.md,
    borderRadius: BorderRadius.full, backgroundColor: Colors.surfaceAlt,
  },
  subNavBtnActive: { backgroundColor: Colors.primaryLight },
  subNavText: {
    fontSize: FontSizes.sm, color: Colors.textMuted, fontWeight: '500',
    marginLeft: 4,
  },
  subNavTextActive: { color: Colors.primary, fontWeight: '600' },
  // Screen
  screenContainer: { flex: 1 },
});

export default StudentApp;
