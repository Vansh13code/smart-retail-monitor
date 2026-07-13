import DashboardLayout from "../layouts/DashboardLayout";
import AnalysisPage from "../components/AnalysisPage";

export default function Reports(){

return(

<DashboardLayout>

<AnalysisPage
  title="Generate Report"
  endpoint="/generate-report/"
/>

</DashboardLayout>

)

}
