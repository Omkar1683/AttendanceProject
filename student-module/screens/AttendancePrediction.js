/**
 * student-module/screens/AttendancePrediction.js
 * --------------------------------------------------
 * Feature 7: Attendance prediction calculator.
 * Shows "what if" scenarios — how many classes to attend for target %.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, RefreshControl, TextInput,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { DashboardSkeleton } from '../components/LoadingSkeleton';
import ErrorState from '../components/ErrorState';
import ProgressBar from '../components/ProgressBar';
import studentApi from '../services/studentApi';
import {
  calculatePercentage,
  percentageAfterMissing,
  classesNeededForTarget,
  generatePredictions,
} from '../utils/attendanceCalculations';

const AttendancePrediction = () => {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [customTarget, setCustomTarget] = useState('75');
  const [missCount, setMissCount] = useState('3');

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const res = await studentApi.getReport();
      if (res.status === 'success') setReport(res.data);
    } catch (err) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const onRefresh = useCallback(() => { setRefreshing(true); loadData(); }, [loadData]);

  if (loading) return <DashboardSkeleton />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;

  const totalPresent = report?.total_present || 0;
  const totalClasses = report?.total_classes || 0;
  const currentPct = report?.overall_percentage || 0;
  const subjects = report?.subjects || [];

  // Standard predictions
  const predictions = generatePredictions(totalPresent, totalClasses);

  // Custom target calculation
  const customTargetNum = parseInt(customTarget, 10) || 75;
  const customNeeded = classesNeededForTarget(totalPresent, totalClasses, customTargetNum);

  // "What if I miss N classes?" calculation
  const missCountNum = parseInt(missCount, 10) || 0;
  const afterMissingPct = percentageAfterMissing(totalPresent, totalClasses, missCountNum);

  return (
    <ScrollView
      style={predStyles.container}
      contentContainerStyle={predStyles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
      showsVerticalScrollIndicator={false}
    >
      {/* Current Status */}
      <View style={predStyles.card}>
        <Text style={predStyles.cardLabel}>CURRENT STATUS</Text>
        <View style={predStyles.statusRow}>
          <View>
            <Text style={[predStyles.bigPct, {
              color: currentPct >= 75 ? Colors.success : Colors.danger,
            }]}>
              {currentPct}%
            </Text>
            <Text style={predStyles.detail}>{totalPresent} / {totalClasses} classes attended</Text>
          </View>
          <View style={[predStyles.statusPill, {
            backgroundColor: currentPct >= 75 ? Colors.successLight : Colors.dangerLight,
          }]}>
            <Text style={{
              color: currentPct >= 75 ? Colors.success : Colors.danger,
              fontWeight: 'bold', fontSize: FontSizes.sm,
            }}>
              {currentPct >= 75 ? '✅ Safe' : '⚠ At Risk'}
            </Text>
          </View>
        </View>
        <ProgressBar percentage={currentPct} height={10} style={{ marginTop: Spacing.md }} />
      </View>

      {/* Standard Target Predictions */}
      <View style={predStyles.card}>
        <Text style={predStyles.cardLabel}>CLASSES NEEDED TO REACH</Text>
        {predictions.map((pred, i) => (
          <View key={pred.target} style={predStyles.predRow}>
            <View style={predStyles.predLeft}>
              <Text style={predStyles.predTarget}>{pred.target}%</Text>
              <ProgressBar
                percentage={pred.achieved ? 100 : Math.min((currentPct / pred.target) * 100, 100)}
                height={6}
                color={pred.achieved ? Colors.success : Colors.primary}
                style={{ flex: 1, marginLeft: Spacing.md }}
              />
            </View>
            <View style={predStyles.predRight}>
              {pred.achieved ? (
                <View style={[predStyles.predBadge, { backgroundColor: Colors.successLight }]}>
                  <Text style={{ color: Colors.success, fontWeight: 'bold', fontSize: FontSizes.sm }}>
                    ✅ Achieved
                  </Text>
                </View>
              ) : pred.impossible ? (
                <Text style={predStyles.predImpossible}>Impossible</Text>
              ) : (
                <View style={[predStyles.predBadge, { backgroundColor: Colors.primaryLight }]}>
                  <Text style={{ color: Colors.primary, fontWeight: 'bold', fontSize: FontSizes.sm }}>
                    {pred.needed} more
                  </Text>
                </View>
              )}
            </View>
          </View>
        ))}
      </View>

      {/* What-If: Miss Classes */}
      <View style={predStyles.card}>
        <Text style={predStyles.cardLabel}>WHAT IF I MISS...</Text>
        <View style={predStyles.inputRow}>
          <Text style={predStyles.inputLabel}>If I miss the next</Text>
          <TextInput
            style={predStyles.input}
            value={missCount}
            onChangeText={setMissCount}
            keyboardType="number-pad"
            maxLength={3}
          />
          <Text style={predStyles.inputLabel}>classes</Text>
        </View>
        <View style={predStyles.resultBox}>
          <Text style={predStyles.resultLabel}>Your attendance will drop to:</Text>
          <Text style={[predStyles.resultValue, {
            color: afterMissingPct >= 75 ? Colors.success : Colors.danger,
          }]}>
            {afterMissingPct}%
          </Text>
          <ProgressBar percentage={afterMissingPct} height={8} style={{ marginTop: Spacing.sm }} />
          {afterMissingPct < 75 && currentPct >= 75 && (
            <View style={predStyles.alertBox}>
              <Ionicons name="warning" size={14} color={Colors.danger} />
              <Text style={predStyles.alertText}>
                You'll become a defaulter if you miss {missCountNum} class{missCountNum !== 1 ? 'es' : ''}!
              </Text>
            </View>
          )}
        </View>
      </View>

      {/* Custom Target */}
      <View style={predStyles.card}>
        <Text style={predStyles.cardLabel}>CUSTOM TARGET</Text>
        <View style={predStyles.inputRow}>
          <Text style={predStyles.inputLabel}>Reach</Text>
          <TextInput
            style={predStyles.input}
            value={customTarget}
            onChangeText={setCustomTarget}
            keyboardType="number-pad"
            maxLength={3}
          />
          <Text style={predStyles.inputLabel}>% attendance</Text>
        </View>
        <View style={predStyles.resultBox}>
          {customNeeded === 0 ? (
            <Text style={[predStyles.resultValue, { color: Colors.success }]}>
              ✅ Already achieved!
            </Text>
          ) : customNeeded === null ? (
            <Text style={[predStyles.resultValue, { color: Colors.danger }]}>
              Impossible to reach with current record
            </Text>
          ) : (
            <Text style={predStyles.resultValue}>
              Attend <Text style={{ color: Colors.primary, fontWeight: 'bold' }}>{customNeeded}</Text> more consecutive classes
            </Text>
          )}
        </View>
      </View>

      {/* Per-Subject Predictions */}
      {subjects.length > 0 && (
        <View style={predStyles.card}>
          <Text style={predStyles.cardLabel}>PER-SUBJECT PROJECTIONS (75%)</Text>
          {subjects.map((subj, i) => {
            const needed = classesNeededForTarget(
              subj.present || 0,
              subj.total_classes || 0,
              75
            );
            return (
              <View key={subj.class_id || i} style={predStyles.subjRow}>
                <View style={{ flex: 1 }}>
                  <Text style={predStyles.subjName}>{subj.name}</Text>
                  <Text style={predStyles.subjDetail}>
                    {subj.present}/{subj.total_classes} • {subj.percentage}%
                  </Text>
                </View>
                {needed === 0 ? (
                  <Text style={[predStyles.subjNeeded, { color: Colors.success }]}>✅ Safe</Text>
                ) : needed === null ? (
                  <Text style={[predStyles.subjNeeded, { color: Colors.textMuted }]}>—</Text>
                ) : (
                  <Text style={[predStyles.subjNeeded, { color: Colors.primary }]}>
                    +{needed} needed
                  </Text>
                )}
              </View>
            );
          })}
        </View>
      )}

      <View style={{ height: 24 }} />
    </ScrollView>
  );
};

const predStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: Spacing.lg },
  card: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.xl,
    padding: Spacing.lg, marginBottom: Spacing.lg,
    borderWidth: 1, borderColor: Colors.border,
  },
  cardLabel: {
    fontSize: FontSizes.xs, fontWeight: 'bold', color: Colors.textMuted,
    marginBottom: Spacing.md, letterSpacing: 1,
  },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  bigPct: { fontSize: 36, fontWeight: 'bold' },
  detail: { fontSize: FontSizes.sm, color: Colors.textTertiary },
  statusPill: { paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm, borderRadius: BorderRadius.full },
  // Predictions
  predRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: Spacing.md, borderBottomWidth: 1, borderBottomColor: Colors.borderLight,
  },
  predLeft: { flexDirection: 'row', alignItems: 'center', flex: 1, marginRight: Spacing.md },
  predTarget: { fontSize: FontSizes.md, fontWeight: 'bold', color: Colors.text, width: 40 },
  predRight: { alignItems: 'flex-end' },
  predBadge: { paddingHorizontal: Spacing.md, paddingVertical: 4, borderRadius: BorderRadius.full },
  predImpossible: { fontSize: FontSizes.sm, color: Colors.textMuted, fontStyle: 'italic' },
  // Input
  inputRow: { flexDirection: 'row', alignItems: 'center', marginBottom: Spacing.md },
  inputLabel: { fontSize: FontSizes.md, color: Colors.textSecondary },
  input: {
    backgroundColor: Colors.surfaceAlt, borderWidth: 1, borderColor: Colors.border,
    borderRadius: BorderRadius.md, paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm,
    fontSize: FontSizes.lg, fontWeight: 'bold', color: Colors.primary,
    textAlign: 'center', width: 56, marginHorizontal: Spacing.sm,
  },
  resultBox: {
    backgroundColor: Colors.surfaceAlt, borderRadius: BorderRadius.lg,
    padding: Spacing.lg,
  },
  resultLabel: { fontSize: FontSizes.sm, color: Colors.textTertiary, marginBottom: 4 },
  resultValue: { fontSize: FontSizes.lg, fontWeight: '600', color: Colors.text },
  alertBox: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: Colors.dangerBg, padding: Spacing.sm,
    borderRadius: BorderRadius.md, marginTop: Spacing.sm,
  },
  alertText: { color: Colors.danger, fontSize: FontSizes.xs, marginLeft: Spacing.xs, flex: 1 },
  // Subject
  subjRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: Spacing.md, borderBottomWidth: 1, borderBottomColor: Colors.borderLight,
  },
  subjName: { fontSize: FontSizes.md, fontWeight: '600', color: Colors.text },
  subjDetail: { fontSize: FontSizes.xs, color: Colors.textMuted, marginTop: 2 },
  subjNeeded: { fontSize: FontSizes.sm, fontWeight: '600' },
});

export default AttendancePrediction;
