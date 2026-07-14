import DashboardLayout from '../layouts/DashboardLayout';
import AnalysisPage from '../components/AnalysisPage';

export default function Pipeline() {
  return (
    <DashboardLayout>
      <AnalysisPage title="Full Pipeline" endpoint="/complete-pipeline/" />
    </DashboardLayout>
  );
}
