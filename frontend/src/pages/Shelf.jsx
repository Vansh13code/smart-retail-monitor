import DashboardLayout from "../layouts/DashboardLayout";
import AnalysisPage from "../components/AnalysisPage";

export default function Shelf(){

return(

<DashboardLayout>

<AnalysisPage
  title="Shelf Detection"
  endpoint="/detect-shelf/"
/>

</DashboardLayout>

)

}
