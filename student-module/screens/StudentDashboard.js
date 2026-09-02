/**
 * student-module/screens/StudentDashboard.js
 * --------------------------------------------
 * Feature 1: Student Dashboard
 * Shows welcome section, attendance summary, quick stats, and subject highlights.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  RefreshControl, ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { DashboardSkeleton } from '../components/LoadingSkeleton';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import ProgressBar from '../components/ProgressBar';
import StatusBadge from '../components/StatusBadge';
import SubjectCard from '../components/SubjectCard';
import studentApi from '../services/studentApi';
import { getAttendanceStatus } from '../utils/attendanceCalculations';

const StudentDashboard = ({ userInfo, navigateTo }) => {
  const [report, setReport] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [reportResult, profileResult] = await Promise.all([
        studentApi.getReport(),
        studentApi.getProfile(),
      ]);

      if (reportResult.status === 'success') setReport(reportResult.data);
      if (profileResult.status === 'success') setProfile(profileResult.data);
    } catch (err) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData();
  }, [loadData]);

  if (loading) return <DashboardSkeleton />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;

  const overallPct = report?.overall_percentage || 0;
  const totalPresent = report?.total_present || 0;
  const totalClasses = report?.total_classes || 0;
  const totalAbsent = totalClasses - totalPresent;
  const subjects = report?.subjects || [];
  const lowSubjects = subjects.filter(s => s.percentage < 75);

  const displayName = profile?.name || userInfo?.name || 'Student';
  const rollNo = profile?.roll_no || userInfo?.roll_no || '';
  const department = profile?.department || userInfo?.department || '';
  const batch = profile?.batch || userInfo?.batch || '';

  return (
    <ScrollView
      style={dashStyles.container}
      contentContainerStyle={dashStyles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
      showsVerticalScrollIndicator={false}
    >
      {/* ── Welcome Card ───────────────────────────────────────────────── */}
      <View style={dashStyles.welcomeCard}>
        <View style={dashStyles.welcomeRow}>
          <View style={dashStyles.avatar}>
            <Text style={dashStyles.avatarText}>
              {displayName.charAt(0).toUpperCase()}
            </Text>
          </View>
          <View style={dashStyles.welcomeInfo}>
            <Text style={dashStyles.welcomeName}>Hi, {displayName.split(' ')[0]}! 👋</Text>
            <Text style={dashStyles.welcomeDetail}>
              {rollNo ? `${rollNo}` : ''}{rollNo && department ? ' • ' : ''}{department}
            </Text>
            {batch ? <Text style={dashStyles.welcomeDetail}>{batch}</Text> : null}
          </View>
        </View>
      </View>

      {/* ── Overall Attendance ─────────────────────────────────────────── */}
      <View style={dashStyles.card}>
        <Text style={dashStyles.cardLabel}>OVERALL ATTENDANCE</Text>
        <View style={dashStyles.overallRow}>
          <View>
            <Text style={[dashStyles.bigPercentage, {
              color: overallPct >= 75 ? Colors.success : Colors.danger,
            }]}>
              {overallPct}%
            </Text>
            <StatusBadge percentage={overallPct} size="lg" />
          </View>
          <View style={dashStyles.overallRight}>
            <Text style={dashStyles.overallDetail}>
              {totalPresent} / {totalClasses} classes
            </Text>
          </View>
        </View>
        <ProgressBar percentage={overallPct} height={10} style={{ marginTop: Spacing.md }} />
        {overallPct < 75 && (
          <View style={dashStyles.warningBanner}>
            <Ionicons name="warning" size={16} color={Colors.danger} />
            <Text style={dashStyles.warningText}>
              Below required minimum (75%). Attend more classes to improve.
            </Text>
          </View>
        )}
      </View>

      {/* ── Quick Stats ────────────────────────────────────────────────── */}
      <View style={dashStyles.statsRow}>
        <View style={[dashStyles.statCard, { borderLeftColor: Colors.success }]}>
          <Text style={dashStyles.statLabel}>Present</Text>
          <Text style={[dashStyles.statValue, { color: Colors.success }]}>{totalPresent}</Text>
        </View>
        <View style={[dashStyles.statCard, { borderLeftColor: Colors.danger }]}>
          <Text style={dashStyles.statLabel}>Absent</Text>
          <Text style={[dashStyles.statValue, { color: Colors.danger }]}>{totalAbsent}</Text>
        </View>
        <View style={[dashStyles.statCard, { borderLeftColor: Colors.warning }]}>
          <Text style={dashStyles.statLabel}>Low (&lt;75%)</Text>
          <Text style={[dashStyles.statValue, { color: Colors.warning }]}>{lowSubjects.length}</Text>
        </View>
      </View>

      {/* ── Subject Breakdown ──────────────────────────────────────────── */}
      {subjects.length > 0 && (
        <>
          <View style={dashStyles.sectionHeader}>
            <Text style={dashStyles.sectionTitle}>Subjects</Text>
            <TouchableOpacity onPress={() => navigateTo('SubjectAttendance')}>
              <Text style={dashStyles.seeAll}>View All →</Text>
            </TouchableOpacity>
          </View>
          {subjects.slice(0, 4).map((subject, index) => (
            <SubjectCard
              key={subject.class_id || index}
              subject={subject}
              onPress={() => navigateTo('SubjectAttendance')}
            />
          ))}
        </>
      )}

      {subjects.length === 0 && (
        <EmptyState
          icon="📚"
          title="No Subjects Yet"
          subtitle="You haven't been enrolled in any classes yet. Contact your teacher."
        />
      )}

      {/* ── Quick Actions ──────────────────────────────────────────────── */}
      <View style={dashStyles.quickActions}>
        <TouchableOpacity style={dashStyles.actionBtn} onPress={() => navigateTo('AttendanceCalendar')}>
          <Ionicons name="calendar" size={20} color={Colors.primary} />
          <Text style={dashStyles.actionText}>Calendar</Text>
        </TouchableOpacity>
        <TouchableOpacity style={dashStyles.actionBtn} onPress={() => navigateTo('AttendancePrediction')}>
          <Ionicons name="trending-up" size={20} color={Colors.accent} />
          <Text style={dashStyles.actionText}>Predict</Text>
        </TouchableOpacity>
        <TouchableOpacity style={dashStyles.actionBtn} onPress={() => navigateTo('ExportAttendance')}>
          <Ionicons name="download-outline" size={20} color={Colors.success} />
          <Text style={dashStyles.actionText}>Export</Text>
        </TouchableOpacity>
        <TouchableOpacity style={dashStyles.actionBtn} onPress={() => navigateTo('AttendanceHistory')}>
          <Ionicons name="time-outline" size={20} color={Colors.orange} />
          <Text style={dashStyles.actionText}>History</Text>
        </TouchableOpacity>
      </View>

      <View style={{ height: 24 }} />
    </ScrollView>
  );
};

const dashStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: Spacing.lg },
  // Welcome
  welcomeCard: {
    backgroundColor: Colors.accent,
    borderRadius: BorderRadius.xxl,
    padding: Spacing.xl,
    marginBottom: Spacing.lg,
  },
  welcomeRow: { flexDirection: 'row', alignItems: 'center' },
  avatar: {
    width: 50, height: 50, borderRadius: 25,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  avatarText: { color: Colors.textWhite, fontSize: 20, fontWeight: 'bold' },
  welcomeInfo: { marginLeft: Spacing.md, flex: 1 },
  welcomeName: { color: Colors.textWhite, fontSize: FontSizes.xl, fontWeight: 'bold' },
  welcomeDetail: { color: 'rgba(255,255,255,0.8)', fontSize: FontSizes.sm, marginTop: 2 },
  // Overall
  card: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.xl,
    padding: Spacing.lg,
    marginBottom: Spacing.lg,
    borderWidth: 1, borderColor: Colors.border,
  },
  cardLabel: {
    fontSize: FontSizes.xs, fontWeight: 'bold',
    color: Colors.textMuted, marginBottom: Spacing.sm,
    letterSpacing: 1,
  },
  overallRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  bigPercentage: { fontSize: 36, fontWeight: 'bold', marginBottom: 4 },
  overallRight: { alignItems: 'flex-end' },
  overallDetail: { fontSize: FontSizes.sm, color: Colors.textTertiary },
  warningBanner: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: Colors.dangerBg, padding: Spacing.md,
    borderRadius: BorderRadius.md, marginTop: Spacing.md,
  },
  warningText: { color: Colors.danger, fontSize: FontSizes.xs, marginLeft: Spacing.sm, flex: 1, fontWeight: '500' },
  // Stats Row
  statsRow: { flexDirection: 'row', gap: Spacing.sm, marginBottom: Spacing.lg },
  statCard: {
    flex: 1, backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg, padding: Spacing.md,
    borderWidth: 1, borderColor: Colors.border,
    borderLeftWidth: 3, alignItems: 'center',
  },
  statLabel: { fontSize: FontSizes.xs, color: Colors.textMuted, fontWeight: '600', marginBottom: 4 },
  statValue: { fontSize: FontSizes.xxl, fontWeight: 'bold' },
  // Section Header
  sectionHeader: {
    flexDirection: 'row', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: Spacing.md,
  },
  sectionTitle: { fontSize: FontSizes.lg, fontWeight: 'bold', color: Colors.text },
  seeAll: { fontSize: FontSizes.sm, color: Colors.primary, fontWeight: '600' },
  // Quick Actions
  quickActions: {
    flexDirection: 'row', justifyContent: 'space-between',
    marginTop: Spacing.lg, gap: Spacing.sm,
  },
  actionBtn: {
    flex: 1, backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg, padding: Spacing.md,
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: Colors.border,
    minHeight: 70,
  },
  actionText: {
    fontSize: FontSizes.xs, color: Colors.textSecondary,
    fontWeight: '600', marginTop: Spacing.xs,
  },
});

export default StudentDashboard;
