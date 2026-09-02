/**
 * student-module/screens/SubjectAttendance.js
 * ---------------------------------------------
 * Feature 2: Subject-wise attendance with detailed cards.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, RefreshControl,
} from 'react-native';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { DashboardSkeleton } from '../components/LoadingSkeleton';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import SubjectCard from '../components/SubjectCard';
import studentApi from '../services/studentApi';

const SubjectAttendance = ({ navigateTo }) => {
  const [report, setReport] = useState(null);
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [reportRes, classesRes] = await Promise.all([
        studentApi.getReport(),
        studentApi.getEnrolledClasses(),
      ]);
      if (reportRes.status === 'success') setReport(reportRes.data);
      if (classesRes.status === 'success') setClasses(classesRes.data || []);
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

  const subjects = report?.subjects || [];

  // Merge teacher info from classes
  const classMap = {};
  classes.forEach(c => { classMap[c.class_id] = c; });
  const enrichedSubjects = subjects.map(s => ({
    ...s,
    teacher_name: classMap[s.class_id]?.teacher_name || '',
  }));

  const safeSubjects = enrichedSubjects.filter(s => s.percentage >= 75);
  const warningSubjects = enrichedSubjects.filter(s => s.percentage < 75);

  return (
    <ScrollView
      style={subjStyles.container}
      contentContainerStyle={subjStyles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
      showsVerticalScrollIndicator={false}
    >
      {/* Summary */}
      <View style={subjStyles.summaryRow}>
        <View style={[subjStyles.summaryCard, { backgroundColor: Colors.successBg }]}>
          <Text style={[subjStyles.summaryValue, { color: Colors.success }]}>{safeSubjects.length}</Text>
          <Text style={subjStyles.summaryLabel}>Safe (≥75%)</Text>
        </View>
        <View style={[subjStyles.summaryCard, { backgroundColor: Colors.dangerBg }]}>
          <Text style={[subjStyles.summaryValue, { color: Colors.danger }]}>{warningSubjects.length}</Text>
          <Text style={subjStyles.summaryLabel}>Below 75%</Text>
        </View>
      </View>

      {/* Warning Subjects */}
      {warningSubjects.length > 0 && (
        <>
          <View style={subjStyles.sectionHeader}>
            <Text style={[subjStyles.sectionTitle, { color: Colors.danger }]}>⚠ Needs Attention</Text>
          </View>
          {warningSubjects.map((subject, index) => (
            <View key={subject.class_id || index}>
              <SubjectCard subject={subject} />
              {subject.teacher_name ? (
                <Text style={subjStyles.teacherName}>👨‍🏫 {subject.teacher_name}</Text>
              ) : null}
            </View>
          ))}
        </>
      )}

      {/* Safe Subjects */}
      {safeSubjects.length > 0 && (
        <>
          <View style={subjStyles.sectionHeader}>
            <Text style={[subjStyles.sectionTitle, { color: Colors.success }]}>✅ On Track</Text>
          </View>
          {safeSubjects.map((subject, index) => (
            <View key={subject.class_id || index}>
              <SubjectCard subject={subject} />
              {subject.teacher_name ? (
                <Text style={subjStyles.teacherName}>👨‍🏫 {subject.teacher_name}</Text>
              ) : null}
            </View>
          ))}
        </>
      )}

      {enrichedSubjects.length === 0 && (
        <EmptyState
          icon="📭"
          title="No Subject Data"
          subtitle="No attendance has been recorded yet."
        />
      )}

      <View style={{ height: 24 }} />
    </ScrollView>
  );
};

const subjStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: Spacing.lg },
  summaryRow: { flexDirection: 'row', gap: Spacing.sm, marginBottom: Spacing.lg },
  summaryCard: {
    flex: 1, borderRadius: BorderRadius.lg, padding: Spacing.lg,
    alignItems: 'center',
  },
  summaryValue: { fontSize: 28, fontWeight: 'bold' },
  summaryLabel: { fontSize: FontSizes.xs, color: Colors.textTertiary, fontWeight: '600', marginTop: 4 },
  sectionHeader: { marginBottom: Spacing.md, marginTop: Spacing.sm },
  sectionTitle: { fontSize: FontSizes.lg, fontWeight: 'bold' },
  teacherName: {
    fontSize: FontSizes.xs, color: Colors.textTertiary,
    marginTop: -6, marginBottom: Spacing.sm, marginLeft: Spacing.lg,
  },
});

export default SubjectAttendance;
