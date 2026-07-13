import DashboardLayout from "../layouts/DashboardLayout";
import AnalysisPage from "../components/AnalysisPage";

export default function Pipeline(){

return(

<DashboardLayout>

<AnalysisPage
  title="Complete Pipeline"
  endpoint="/complete-pipeline/"
/>

</DashboardLayout>

)

}
